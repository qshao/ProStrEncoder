# ProStrEncoder v2 Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scale ProStrEncoder from 2.2M to ~85M–2.5B parameters by adding a global Flash Attention transformer after the GVP local encoder, dual pre-training objectives (masked residue + inverse folding), two-phase training with layer-wise LR, DDP/FSDP multi-GPU support, and a quality-filtered PDB data pipeline.

**Architecture:** GVP-GNN local encoder (existing, scaled up) → bridge Linear → GlobalTransformer (Flash Attention 2, RoPE, sliding window) → per-residue embeddings. Alternating batches: 70% masked residue prediction, 30% inverse folding. Two-phase training: GVP warm-up for 5% of steps, then joint training with layer-wise LR (GVP at 0.1×).

**Tech Stack:** PyTorch 2.0+, PyTorch Geometric, flash-attn≥2.5.0 (dao-ailab), webdataset, torchrun for multi-GPU, MMseqs2 for train/val clustering, BF16 precision throughout.

**Design doc:** `docs/superpowers/specs/2026-06-02-prostrencoder-v2-scaling-design.md`

---

## Dependency Order

- **Group A (Tasks 1–3):** Model architecture. Sequential — each task depends on the previous.
- **Group B (Tasks 4–5):** Training objectives and trainer. Depends on Group A Task 2.
- **Group C (Task 6):** Multi-GPU train script. Depends on Group B Task 5.
- **Group D (Tasks 7–8):** Data pipeline. Fully independent — can start in parallel with Group A.
- **Group E (Task 9):** Evaluation. Depends on Groups A–B.

---

## File Map

**New files:**
- `prostrencoder/models/global_transformer.py` — `GlobalTransformerLayer`, `GlobalTransformer`
- `tests/models/test_global_transformer.py`
- `configs/medium.yaml` — 85M param tier
- `configs/large.yaml` — 860M param tier
- `configs/xl.yaml` — 2.5B param tier
- `scripts/download_pdb.py` — RCSB download + per-chain quality filtering
- `scripts/eval_inverse_folding.py` — sequence recovery evaluation

**Modified files:**
- `prostrencoder/models/encoder.py` — add GlobalTransformer, bridge, `_transformer_forward`, `hidden_dim`, `use_transformer` flag
- `prostrencoder/training/objectives.py` — add `InverseFoldingLoss`, `sample_batch_mode`
- `prostrencoder/training/trainer.py` — two-phase schedule, layer-wise LR, alternating batch, BF16, gradient checkpointing, JSON logging
- `scripts/train.py` — device strategy (single_gpu/ddp/fsdp), `torchrun` validation
- `scripts/preprocess.py` — parallel preprocessing with `multiprocessing.Pool`
- `configs/default.yaml` — add new v2 config keys with backward-compat defaults
- `tests/models/test_encoder.py` — add v2 (tx_layers>0) tests
- `tests/training/test_objectives.py` — add InverseFoldingLoss tests

---

## Group A — Model Architecture

---

### Task 1: GlobalTransformer Module

**Files:**
- Create: `prostrencoder/models/global_transformer.py`
- Create: `tests/models/test_global_transformer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/models/test_global_transformer.py
import pytest
import torch

try:
    import flash_attn  # noqa: F401
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def _make_layer(d_model=64, num_heads=4, window_size=-1):
    from prostrencoder.models.global_transformer import GlobalTransformerLayer
    return GlobalTransformerLayer(d_model=d_model, num_heads=num_heads,
                                  ff_mult=4, dropout=0.0,
                                  window_size=window_size)


def _make_transformer(d_in=144, d_model=64, num_layers=2, num_heads=4):
    from prostrencoder.models.global_transformer import GlobalTransformer
    return GlobalTransformer(d_in=d_in, d_model=d_model, num_layers=num_layers,
                             num_heads=num_heads, ff_mult=4, dropout=0.0)


def test_layer_output_shape():
    layer = _make_layer()
    x = torch.randn(2, 10, 64)   # (B, N, D)
    out = layer(x)
    assert out.shape == (2, 10, 64)


def test_layer_with_padding_mask():
    layer = _make_layer()
    x = torch.randn(2, 10, 64)
    # Last 3 positions of second sequence are padding
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[1, 7:] = True
    out = layer(x, key_padding_mask=mask)
    assert out.shape == (2, 10, 64)
    assert torch.isfinite(out).all()


def test_transformer_bridges_input_dim():
    """Bridge must project d_in → d_model correctly."""
    model = _make_transformer(d_in=144, d_model=64)
    x = torch.randn(2, 12, 144)   # (B, N, d_in)
    out = model(x)
    assert out.shape == (2, 12, 64)


def test_transformer_output_finite():
    model = _make_transformer()
    x = torch.randn(3, 8, 144)
    out = model(x)
    assert torch.isfinite(out).all()


def test_transformer_gradient_checkpointing():
    from prostrencoder.models.global_transformer import GlobalTransformer
    model = GlobalTransformer(d_in=144, d_model=64, num_layers=2,
                              num_heads=4, ff_mult=4, dropout=0.0,
                              gradient_checkpointing=True)
    x = torch.randn(2, 8, 144, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None


@pytest.mark.skipif(not HAS_FLASH_ATTN, reason="flash_attn not installed")
def test_flash_attn_layer_matches_fallback_shape():
    """Flash and fallback layers produce the same output shape."""
    from prostrencoder.models.global_transformer import GlobalTransformerLayer
    x = torch.randn(2, 16, 64)
    layer = GlobalTransformerLayer(64, 4, dropout=0.0)
    out = layer(x)
    assert out.shape == (2, 16, 64)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/models/test_global_transformer.py -v 2>&1 | head -20
```
Expected: `ImportError: cannot import name 'GlobalTransformerLayer'`

- [ ] **Step 3: Implement `global_transformer.py`**

```python
# prostrencoder/models/global_transformer.py
import warnings
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as ckpt

try:
    from flash_attn.modules.mha import MHA as FlashMHA
    from flash_attn.modules.mlp import GatedMlp
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    warnings.warn(
        "flash_attn not installed — falling back to torch.nn.MultiheadAttention. "
        "Install with: pip install flash-attn --no-build-isolation",
        stacklevel=2,
    )


class GlobalTransformerLayer(nn.Module):
    """
    Single pre-norm transformer layer.  Uses Flash Attention 2 when available
    (dao-ailab/flash-attention); falls back to torch.nn.MultiheadAttention.

    Args:
        d_model     : embedding dimension
        num_heads   : attention heads (must divide d_model)
        ff_mult     : feed-forward hidden = d_model * ff_mult
        dropout     : attention + FF dropout rate
        window_size : sliding window half-size per side; -1 = full attention
    """

    def __init__(self, d_model: int, num_heads: int, ff_mult: int = 4,
                 dropout: float = 0.1, window_size: int = -1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if HAS_FLASH_ATTN:
            self.attn = FlashMHA(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                causal=False,
                window_size=(window_size, window_size),
                rotary_emb_dim=d_model // num_heads,
            )
            self.ff = GatedMlp(
                in_features=d_model,
                hidden_features=d_model * ff_mult,
                activation=nn.functional.silu,
            )
            self._use_flash = True
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_model * ff_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * ff_mult, d_model),
                nn.Dropout(dropout),
            )
            self._use_flash = False

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x               : (B, N, d_model)
        key_padding_mask: (B, N) bool — True = padding position (ignored)
        Returns         : (B, N, d_model)
        """
        if self._use_flash:
            x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        else:
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed,
                                    key_padding_mask=key_padding_mask)
            x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class GlobalTransformer(nn.Module):
    """
    Stack of GlobalTransformerLayer blocks with an input bridge projection.

    Converts flat GVP output (N, d_in) → padded (B, max_N, d_in) → bridge
    → transformer layers → (B, max_N, d_model).  The bridge and unpacking
    are handled externally by ProStrEncoder._transformer_forward.

    Args:
        d_in                  : input feature dimension from GVP
        d_model               : transformer hidden dimension
        num_layers            : number of transformer layers
        num_heads             : attention heads
        ff_mult               : feed-forward multiplier
        dropout               : dropout rate
        window_size           : sliding window per side; -1 = full attention
        gradient_checkpointing: recompute activations on backward (saves memory)
    """

    def __init__(self, d_in: int, d_model: int, num_layers: int,
                 num_heads: int, ff_mult: int = 4, dropout: float = 0.1,
                 window_size: int = -1, gradient_checkpointing: bool = False):
        super().__init__()
        self.d_model = d_model
        self.gradient_checkpointing = gradient_checkpointing

        self.bridge = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([
            GlobalTransformerLayer(d_model, num_heads, ff_mult,
                                   dropout, window_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x               : (B, N, d_in)
        key_padding_mask: (B, N) bool — True = padding
        Returns         : (B, N, d_model)
        """
        x = self.bridge(x)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = ckpt(layer, x, key_padding_mask, use_reentrant=False)
            else:
                x = layer(x, key_padding_mask)
        return self.norm(x)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/models/test_global_transformer.py -v
```
Expected: all tests pass (flash_attn test skipped if not installed)

- [ ] **Step 5: Commit**

```bash
git add prostrencoder/models/global_transformer.py tests/models/test_global_transformer.py
git commit -m "feat: GlobalTransformer — Flash Attention 2 + RoPE + sliding window, SDPA fallback"
```

---

### Task 2: ProStrEncoder v2 — Bridge + Transformer Integration

**Files:**
- Modify: `prostrencoder/models/encoder.py`
- Modify: `tests/models/test_encoder.py`

- [ ] **Step 1: Write failing tests for v2 encoder**

Add these tests to `tests/models/test_encoder.py`:

```python
# Add at the bottom of tests/models/test_encoder.py

V2_CONFIG = {
    "num_layers": 2,
    "node_scalar_in": 91,   # 27 + walk_embed_dim(64)
    "node_vector_in": 3,
    "edge_scalar_in": 16,
    "edge_vector_in": 1,
    "hidden_scalar": 64,
    "hidden_vector": 8,
    "output_dim": 128,
    "dropout": 0.0,
    "num_walks": 4,
    "walk_length": 5,
    "walk_embed_dim": 64,
    "walk_layers": 2,
    "walk_heads": 4,
    # v2 transformer fields
    "tx_layers": 2,
    "tx_d_model": 128,
    "tx_num_heads": 4,
    "tx_ff_mult": 2,
    "tx_dropout": 0.0,
    "attn_window_size": -1,
    "gradient_checkpointing": False,
}


def _fake_batch_v2(n=20, e=40):
    from torch_geometric.data import Data
    return Data(
        seq_idx=torch.randint(0, 20, (n,)),
        x_scalar=torch.randn(n, 27),
        x_vec=torch.randn(n, 3, 3),
        edge_index=torch.randint(0, n, (2, e)),
        edge_scalar=torch.randn(e, 16),
        edge_vec=torch.randn(e, 1, 3),
        batch=torch.zeros(n, dtype=torch.long),   # all in one protein
    )


def test_v2_encoder_output_shape():
    from prostrencoder.models.encoder import ProStrEncoder
    model = ProStrEncoder(V2_CONFIG)
    batch = _fake_batch_v2(n=20)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (20, 128), f"Expected (20, 128), got {out.shape}"


def test_v2_encoder_hidden_dim_property():
    from prostrencoder.models.encoder import ProStrEncoder
    model = ProStrEncoder(V2_CONFIG)
    assert model.hidden_dim == 128   # tx_d_model


def test_v2_encoder_hidden_shape():
    from prostrencoder.models.encoder import ProStrEncoder
    model = ProStrEncoder(V2_CONFIG)
    batch = _fake_batch_v2(n=20)
    with torch.no_grad():
        out, hidden = model(batch, return_hidden=True)
    assert hidden.shape == (20, 128)   # (N, tx_d_model)


def test_v2_encoder_warmup_phase_skips_transformer_layers():
    """In phase 1 (use_transformer=False), bridge runs but not the layers."""
    from prostrencoder.models.encoder import ProStrEncoder
    model = ProStrEncoder(V2_CONFIG)
    model.use_transformer = False
    batch = _fake_batch_v2(n=20)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (20, 128)


def test_v2_encoder_multi_protein_batch():
    """batch.batch with two proteins of different lengths."""
    from prostrencoder.models.encoder import ProStrEncoder
    from torch_geometric.data import Data, Batch
    model = ProStrEncoder(V2_CONFIG)
    d1 = Data(seq_idx=torch.randint(0, 20, (12,)),
              x_scalar=torch.randn(12, 27),
              x_vec=torch.randn(12, 3, 3),
              edge_index=torch.randint(0, 12, (2, 24)),
              edge_scalar=torch.randn(24, 16),
              edge_vec=torch.randn(24, 1, 3))
    d2 = Data(seq_idx=torch.randint(0, 20, (8,)),
              x_scalar=torch.randn(8, 27),
              x_vec=torch.randn(8, 3, 3),
              edge_index=torch.randint(0, 8, (2, 16)),
              edge_scalar=torch.randn(16, 16),
              edge_vec=torch.randn(16, 1, 3))
    batch = Batch.from_data_list([d1, d2])
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (20, 128)   # 12 + 8 = 20 residues


def test_v1_config_still_works():
    """tx_layers=0 must produce the same behavior as before v2."""
    from prostrencoder.models.encoder import ProStrEncoder
    v1_cfg = {
        "num_layers": 2, "node_scalar_in": 91, "node_vector_in": 3,
        "edge_scalar_in": 16, "edge_vector_in": 1,
        "hidden_scalar": 64, "hidden_vector": 8, "output_dim": 128,
        "dropout": 0.0, "num_walks": 4, "walk_length": 5,
        "walk_embed_dim": 64, "walk_layers": 2, "walk_heads": 4,
        # No tx_* keys at all → backward compat
    }
    model = ProStrEncoder(v1_cfg)
    assert model.transformer is None
    assert model.hidden_dim == 64 + 8   # s_h + v_h
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/models/test_encoder.py::test_v2_encoder_output_shape -v 2>&1 | tail -5
```
Expected: `AttributeError: 'ProStrEncoder' object has no attribute 'transformer'`

- [ ] **Step 3: Implement encoder v2**

Replace the full content of `prostrencoder/models/encoder.py`:

```python
# prostrencoder/models/encoder.py
import torch
import torch.nn as nn
from prostrencoder.models.gvp import GVP, GVPConv
from prostrencoder.models.walk_sampler import WalkSampler
from prostrencoder.models.walk_encoder import WalkEncoder
from prostrencoder.models.global_transformer import GlobalTransformer


class ProStrEncoder(nn.Module):
    """
    GVP-GNN local encoder + optional global transformer (v2).

    v1 behavior (tx_layers absent or 0): identical to original — GVP output
    directly projected to embedding space.

    v2 behavior (tx_layers > 0): GVP invariant output → bridge Linear →
    GlobalTransformer → embedding projection. Long-range context captured
    by attention; equivariance handled locally by GVP.

    Two-phase training support:
        self.use_transformer = False  →  Phase 1: bridge only, no attention layers
        self.use_transformer = True   →  Phase 2: full transformer

    The `use_transformer` flag is set by the Trainer at the phase transition.
    """

    def __init__(self, config: dict):
        super().__init__()
        s_n  = config["node_scalar_in"]
        v_n  = config["node_vector_in"]
        s_e  = config["edge_scalar_in"]
        v_e  = config["edge_vector_in"]
        s_h  = config["hidden_scalar"]
        v_h  = config["hidden_vector"]
        out  = config["output_dim"]
        L    = config["num_layers"]
        drop = config["dropout"]

        self._s_h = s_h
        self._v_h = v_h

        # ── Walk components ────────────────────────────────────────────────────
        self.walk_sampler = WalkSampler(
            num_walks=config["num_walks"],
            walk_length=config["walk_length"],
            max_deg_cap=config.get("max_deg_cap", 30),
        )
        walk_seq_len = config["walk_length"] + 2
        self.walk_encoder = WalkEncoder(
            vocab_size=23,
            embed_dim=config["walk_embed_dim"],
            num_heads=config["walk_heads"],
            num_layers=config["walk_layers"],
            ff_dim=config["walk_embed_dim"] * 4,
            max_seq_len=walk_seq_len,
            dropout=drop,
        )

        # ── GVP-GNN backbone ──────────────────────────────────────────────────
        self.node_in = GVP((s_n, v_n), (s_h, v_h))
        self.edge_in = GVP((s_e, v_e), (s_h // 4, v_e))
        s_e_h = s_h // 4
        self.layers = nn.ModuleList([
            GVPConv(node_in_dims=(s_h, v_h), edge_in_dims=(s_e_h, v_e),
                    node_out_dims=(s_h, v_h), drop_rate=drop)
            for _ in range(L)
        ])

        # ── Optional global transformer ───────────────────────────────────────
        tx_layers = config.get("tx_layers", 0)
        if tx_layers > 0:
            d_in = s_h + v_h   # GVP invariant dim (scalars + vector norms)
            self.transformer = GlobalTransformer(
                d_in=d_in,
                d_model=config["tx_d_model"],
                num_layers=tx_layers,
                num_heads=config["tx_num_heads"],
                ff_mult=config.get("tx_ff_mult", 4),
                dropout=config.get("tx_dropout", drop),
                window_size=config.get("attn_window_size", -1),
                gradient_checkpointing=config.get("gradient_checkpointing", False),
            )
            proj_in = config["tx_d_model"]
        else:
            self.transformer = None
            proj_in = s_h + v_h

        # Phase flag: set to True by Trainer when Phase 2 begins
        self.use_transformer: bool = False

        # ── Output projection ─────────────────────────────────────────────────
        self.out_proj = nn.Sequential(
            nn.LayerNorm(proj_in),
            nn.Linear(proj_in, out),
            nn.GELU(),
            nn.Linear(out, out),
        )

    @property
    def hidden_dim(self) -> int:
        """Dimension of the hidden representation returned by forward(return_hidden=True)."""
        if self.transformer is not None:
            return self.transformer.d_model
        return self._s_h + self._v_h

    def _transformer_forward(self, gvp_out: torch.Tensor,
                             batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Packs flat node features into padded (B, max_N, D) tensors, runs the
        global transformer, then gathers back to flat (N, tx_d_model).

        gvp_out  : (N, s_h + v_h) — invariant GVP output
        batch_idx: (N,) int64     — protein index per node (from batch.batch)
        Returns  : (N, tx_d_model)
        """
        N = gvp_out.shape[0]
        B = int(batch_idx.max().item()) + 1
        device = gvp_out.device

        counts = torch.bincount(batch_idx, minlength=B)           # (B,)
        max_len = int(counts.max().item())

        cum = torch.zeros(B + 1, dtype=torch.long, device=device)
        cum[1:] = counts.cumsum(0)
        pos = torch.arange(N, device=device) - cum[batch_idx]     # within-protein pos

        # Scatter into padded tensor
        padded = gvp_out.new_zeros(B, max_len, gvp_out.shape[-1])
        padded[batch_idx, pos] = gvp_out

        # key_padding_mask: True = padding (ignored by attention)
        key_padding_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)
        key_padding_mask[batch_idx, pos] = False

        tx_out = self.transformer(padded, key_padding_mask=key_padding_mask)  # (B, max_len, d_model)
        return tx_out[batch_idx, pos]                              # gather back to (N, d_model)

    def forward(self, batch, return_hidden: bool = False):
        """
        batch        : PyG Batch or Data with fields seq_idx, x_scalar, x_vec,
                       edge_index, edge_scalar, edge_vec, and (for v2) batch.batch.
        return_hidden: if True, returns (output, hidden).
        """
        node_v     = batch.x_vec
        edge_index = batch.edge_index
        edge_s     = batch.edge_scalar
        edge_v     = batch.edge_vec

        # Walk embedding
        walk_tokens = self.walk_sampler(edge_index, batch.seq_idx)
        walk_emb    = self.walk_encoder(walk_tokens)
        node_s = torch.cat([batch.x_scalar, walk_emb], dim=-1)

        # GVP-GNN
        node_s, node_v = self.node_in(node_s, node_v)
        edge_s, edge_v = self.edge_in(edge_s, edge_v)
        for layer in self.layers:
            node_s, node_v = layer(node_s, node_v, edge_index, edge_s, edge_v)

        # Invariant GVP output
        v_norms = torch.linalg.vector_norm(node_v, dim=-1)    # (N, v_h)
        gvp_out = torch.cat([node_s, v_norms], dim=-1)        # (N, s_h + v_h)

        if self.transformer is not None:
            if self.use_transformer:
                # Phase 2: full transformer (bridge + attention layers + norm)
                batch_idx = getattr(batch, "batch",
                                    torch.zeros(node_s.shape[0], dtype=torch.long,
                                                device=node_s.device))
                hidden = self._transformer_forward(gvp_out, batch_idx)
            else:
                # Phase 1: bridge only — no attention, no extra compute
                hidden = self.transformer.bridge(gvp_out)
        else:
            hidden = gvp_out

        output = self.out_proj(hidden)
        if return_hidden:
            return output, hidden
        return output
```

- [ ] **Step 4: Run all encoder tests**

```bash
python -m pytest tests/models/test_encoder.py -v
```
Expected: all tests pass including the new v2 tests.

- [ ] **Step 5: Commit**

```bash
git add prostrencoder/models/encoder.py tests/models/test_encoder.py
git commit -m "feat: ProStrEncoder v2 — GlobalTransformer bridge, two-phase use_transformer flag, hidden_dim property"
```

---

### Task 3: Config Tier YAML Files

**Files:**
- Modify: `configs/default.yaml`
- Create: `configs/medium.yaml`, `configs/large.yaml`, `configs/xl.yaml`

- [ ] **Step 1: Add v2 keys to `configs/default.yaml` (backward compat — tx_layers=0)**

Open `configs/default.yaml` and add the following block inside `model:`, after `max_deg_cap`:

```yaml
  # Global transformer (v2) — set tx_layers: 0 to disable (v1 behavior)
  tx_layers: 0
  tx_d_model: 512
  tx_num_heads: 8
  tx_ff_mult: 4
  tx_dropout: 0.1
  attn_window_size: -1     # -1 = full attention; >0 = sliding window per side
  gradient_checkpointing: false
```

Also add to `training:`:
```yaml
  # v2 training
  precision: fp32                # "fp32" | "bf16"  (bf16 requires CUDA + flash_attn)
  gvp_lr_multiplier: 0.1        # GVP LR = lr * this during Phase 2
  warmup_phase_fraction: 0.05   # Phase 1 length as fraction of total steps
  inverse_fold_prob: 0.3        # fraction of batches using inverse folding loss
  inverse_fold_weight: 1.0      # λ — inverse folding loss scale factor
  save_every_steps: 1000
  device_strategy: single_gpu   # "single_gpu" | "ddp" | "fsdp"
```

- [ ] **Step 2: Create `configs/medium.yaml`**

```yaml
# configs/medium.yaml — 85M parameter tier
# Inherits all defaults; overrides model dimensions and device settings.
# Use: python scripts/train.py --config configs/medium.yaml
#      torchrun --nproc_per_node=2 scripts/train.py --config configs/medium.yaml

data:
  pdb_dir: data/pdb
  processed_dir: data/processed
  train_dir: data/processed/train
  val_dir: data/processed/val
  k_neighbors: 30
  max_len: 1024
  min_len: 40
  resolution_cutoff: 3.5
  min_chain_coverage: 0.9
  cluster_seqid: 0.30
  preprocess_workers: 16
  dataset_format: random_access

model:
  num_layers: 6
  node_scalar_in: 155        # 27 + walk_embed_dim(128)
  node_vector_in: 3
  edge_scalar_in: 16
  edge_vector_in: 1
  hidden_scalar: 256
  hidden_vector: 32
  output_dim: 512
  dropout: 0.1
  num_walks: 8
  walk_length: 20
  walk_embed_dim: 128
  walk_layers: 6
  walk_heads: 8
  max_deg_cap: 30
  tx_layers: 12
  tx_d_model: 512
  tx_num_heads: 8
  tx_ff_mult: 4
  tx_dropout: 0.1
  attn_window_size: -1
  gradient_checkpointing: false

training:
  batch_size: 32
  max_epochs: 100
  lr: 3.0e-4
  weight_decay: 0.01
  mask_rate: 0.15
  grad_clip: 1.0
  checkpoint_dir: checkpoints/medium
  log_every: 50
  num_workers: 4
  precision: bf16
  gvp_lr_multiplier: 0.1
  warmup_phase_fraction: 0.05
  inverse_fold_prob: 0.3
  inverse_fold_weight: 1.0
  save_every_steps: 1000
  device_strategy: single_gpu
  wandb: false
  wandb_project: prostrencoder

evaluation:
  probe_epochs: 100
  probe_lr: 0.01
  probe_weight_decay: 0.0001
```

- [ ] **Step 3: Create `configs/large.yaml`**

```yaml
# configs/large.yaml — 860M parameter tier
# Launch: torchrun --nproc_per_node=4 scripts/train.py --config configs/large.yaml

data:
  pdb_dir: data/pdb
  processed_dir: data/processed
  train_dir: data/processed/train
  val_dir: data/processed/val
  k_neighbors: 30
  max_len: 1024
  min_len: 40
  resolution_cutoff: 3.5
  min_chain_coverage: 0.9
  cluster_seqid: 0.30
  preprocess_workers: 16
  dataset_format: random_access

model:
  num_layers: 8
  node_scalar_in: 283        # 27 + walk_embed_dim(256)
  node_vector_in: 3
  edge_scalar_in: 16
  edge_vector_in: 1
  hidden_scalar: 512
  hidden_vector: 64
  output_dim: 1024
  dropout: 0.1
  num_walks: 8
  walk_length: 20
  walk_embed_dim: 256
  walk_layers: 8
  walk_heads: 8
  max_deg_cap: 30
  tx_layers: 24
  tx_d_model: 1024
  tx_num_heads: 16
  tx_ff_mult: 4
  tx_dropout: 0.1
  attn_window_size: -1
  gradient_checkpointing: true

training:
  batch_size: 16
  max_epochs: 100
  lr: 2.0e-4
  weight_decay: 0.01
  mask_rate: 0.15
  grad_clip: 1.0
  checkpoint_dir: checkpoints/large
  log_every: 50
  num_workers: 4
  precision: bf16
  gvp_lr_multiplier: 0.1
  warmup_phase_fraction: 0.05
  inverse_fold_prob: 0.3
  inverse_fold_weight: 1.0
  save_every_steps: 500
  device_strategy: ddp
  wandb: false
  wandb_project: prostrencoder

evaluation:
  probe_epochs: 100
  probe_lr: 0.01
  probe_weight_decay: 0.0001
```

- [ ] **Step 4: Create `configs/xl.yaml`**

```yaml
# configs/xl.yaml — 2.5B parameter tier
# Launch: torchrun --nproc_per_node=4 scripts/train.py --config configs/xl.yaml

data:
  pdb_dir: data/pdb
  processed_dir: data/processed
  train_dir: data/processed/train
  val_dir: data/processed/val
  k_neighbors: 30
  max_len: 1024
  min_len: 40
  resolution_cutoff: 3.5
  min_chain_coverage: 0.9
  cluster_seqid: 0.30
  preprocess_workers: 16
  dataset_format: random_access

model:
  num_layers: 12
  node_scalar_in: 283        # 27 + walk_embed_dim(256)
  node_vector_in: 3
  edge_scalar_in: 16
  edge_vector_in: 1
  hidden_scalar: 512
  hidden_vector: 64
  output_dim: 1536
  dropout: 0.1
  num_walks: 8
  walk_length: 20
  walk_embed_dim: 256
  walk_layers: 8
  walk_heads: 8
  max_deg_cap: 30
  tx_layers: 32
  tx_d_model: 1536
  tx_num_heads: 16
  tx_ff_mult: 4
  tx_dropout: 0.1
  attn_window_size: -1
  gradient_checkpointing: true

training:
  batch_size: 8
  max_epochs: 100
  lr: 1.5e-4
  weight_decay: 0.01
  mask_rate: 0.15
  grad_clip: 1.0
  checkpoint_dir: checkpoints/xl
  log_every: 50
  num_workers: 4
  precision: bf16
  gvp_lr_multiplier: 0.1
  warmup_phase_fraction: 0.05
  inverse_fold_prob: 0.3
  inverse_fold_weight: 1.0
  save_every_steps: 250
  device_strategy: fsdp
  wandb: false
  wandb_project: prostrencoder

evaluation:
  probe_epochs: 100
  probe_lr: 0.01
  probe_weight_decay: 0.0001
```

- [ ] **Step 5: Verify Small config still loads with tx_layers=0**

```bash
python -c "
import yaml
from prostrencoder.models.encoder import ProStrEncoder
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
model = ProStrEncoder(cfg['model'])
assert model.transformer is None, 'v1 config must not create transformer'
print('OK: default.yaml tx_layers=0 gives no transformer, hidden_dim =', model.hidden_dim)
"
```
Expected: `OK: default.yaml tx_layers=0 gives no transformer, hidden_dim = 144`

- [ ] **Step 6: Commit**

```bash
git add configs/default.yaml configs/medium.yaml configs/large.yaml configs/xl.yaml
git commit -m "feat: config tiers — medium (85M), large (860M), xl (2.5B); update default.yaml with v2 keys"
```

---

## Group B — Training Objectives and Trainer

---

### Task 4: Inverse Folding Objective + Alternating Batch Mode

**Files:**
- Modify: `prostrencoder/training/objectives.py`
- Modify: `tests/training/test_objectives.py`

- [ ] **Step 1: Write failing tests**

Add at the bottom of `tests/training/test_objectives.py`:

```python
# --- Inverse folding objective ---

from prostrencoder.training.objectives import InverseFoldingLoss, sample_batch_mode


def test_inverse_folding_loss_shape():
    N = 50
    loss_fn = InverseFoldingLoss()
    logits  = torch.randn(N, 21)
    targets = torch.randint(0, 21, (N,))
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0, "loss must be scalar"
    assert loss.item() > 0


def test_inverse_folding_loss_all_positions():
    """Loss must use all N positions (unlike masked loss which uses ~15%)."""
    N = 100
    loss_fn = InverseFoldingLoss()
    logits  = torch.zeros(N, 21)
    logits[:, 0] = 100.0   # model predicts class 0 for every residue
    targets = torch.zeros(N, dtype=torch.long)   # true label = 0
    loss = loss_fn(logits, targets)
    # Near-zero loss because predictions match targets
    assert loss.item() < 0.01, f"Expected near-zero loss, got {loss.item()}"


def test_inverse_folding_loss_weighted():
    """pLDDT weights zero out low-confidence residues."""
    N = 20
    loss_fn = InverseFoldingLoss()
    logits  = torch.randn(N, 21)
    targets = torch.randint(0, 21, (N,))
    weights = torch.zeros(N)   # all zero → loss must be 0
    loss = loss_fn(logits, targets, weights=weights)
    assert loss.item() == 0.0


def test_sample_batch_mode_returns_a_or_b():
    modes = {sample_batch_mode(0.3) for _ in range(50)}
    assert modes == {"A", "B"}


def test_sample_batch_mode_prob_zero_always_a():
    assert all(sample_batch_mode(0.0) == "A" for _ in range(10))


def test_sample_batch_mode_prob_one_always_b():
    assert all(sample_batch_mode(1.0) == "B" for _ in range(10))
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/training/test_objectives.py::test_inverse_folding_loss_shape -v 2>&1 | tail -5
```
Expected: `ImportError: cannot import name 'InverseFoldingLoss'`

- [ ] **Step 3: Add `InverseFoldingLoss` and `sample_batch_mode` to objectives.py**

Append to the end of `prostrencoder/training/objectives.py`:

```python

def sample_batch_mode(inverse_fold_prob: float) -> str:
    """
    Randomly choose the training mode for a batch.

    Returns "B" (inverse folding) with probability inverse_fold_prob,
    otherwise "A" (masked residue prediction).
    """
    return "B" if torch.rand(1).item() < inverse_fold_prob else "A"


class InverseFoldingLoss(nn.Module):
    """
    Cross-entropy loss over ALL residue positions (no masking).

    Used in Mode B batches: given full 3D structure, predict the complete
    amino acid sequence.

    Forward args:
        logits  : (N, 21) — raw predictions for all residues
        targets : (N,) int64 — true AA indices 0–20
        weights : (N,) float32 or None — per-residue loss weights (e.g. pLDDT/100).
                  Zero weight excludes a residue from the loss entirely.
                  If None, all residues contribute equally.

    Returns scalar loss.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                weights: torch.Tensor = None) -> torch.Tensor:
        if weights is None:
            return F.cross_entropy(logits, targets)

        # Weighted per-residue cross-entropy
        per_residue = F.cross_entropy(logits, targets, reduction="none")  # (N,)
        total_weight = weights.sum()
        if total_weight == 0:
            return logits.new_tensor(0.0)
        return (per_residue * weights).sum() / total_weight
```

- [ ] **Step 4: Run all objective tests**

```bash
python -m pytest tests/training/test_objectives.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add prostrencoder/training/objectives.py tests/training/test_objectives.py
git commit -m "feat: InverseFoldingLoss (weighted per-residue CE) + sample_batch_mode"
```

---

### Task 5: v2 Trainer — Two-Phase, Layer-wise LR, BF16, Alternating Batch

**Files:**
- Modify: `prostrencoder/training/trainer.py`

- [ ] **Step 1: Write trainer smoke test**

Create `tests/training/test_trainer_v2.py`:

```python
# tests/training/test_trainer_v2.py
import torch
import pytest
from torch_geometric.data import Data, Batch


def _toy_dataset(n_proteins=6, n_residues=15, n_edges=30):
    items = []
    for _ in range(n_proteins):
        items.append(Data(
            seq_idx=torch.randint(0, 20, (n_residues,)),
            x_scalar=torch.randn(n_residues, 27),
            x_vec=torch.randn(n_residues, 3, 3),
            edge_index=torch.randint(0, n_residues, (2, n_edges)),
            edge_scalar=torch.randn(n_edges, 16),
            edge_vec=torch.randn(n_edges, 1, 3),
        ))
    return items


V2_CONFIG = {
    "model": {
        "num_layers": 2, "node_scalar_in": 91, "node_vector_in": 3,
        "edge_scalar_in": 16, "edge_vector_in": 1,
        "hidden_scalar": 32, "hidden_vector": 4, "output_dim": 64,
        "dropout": 0.0, "num_walks": 4, "walk_length": 5,
        "walk_embed_dim": 64, "walk_layers": 2, "walk_heads": 4,
        "max_deg_cap": 30,
        "tx_layers": 2, "tx_d_model": 64, "tx_num_heads": 4,
        "tx_ff_mult": 2, "tx_dropout": 0.0, "attn_window_size": -1,
        "gradient_checkpointing": False,
    },
    "training": {
        "batch_size": 2, "max_epochs": 2, "lr": 1e-3,
        "weight_decay": 0.01, "mask_rate": 0.15, "grad_clip": 1.0,
        "checkpoint_dir": "/tmp/test_ckpt_v2", "log_every": 100,
        "num_workers": 0, "precision": "fp32",
        "gvp_lr_multiplier": 0.1, "warmup_phase_fraction": 0.5,
        "inverse_fold_prob": 0.5, "inverse_fold_weight": 1.0,
        "save_every_steps": 999999,
        "device_strategy": "single_gpu",
    },
}


def test_trainer_v2_two_epochs():
    from prostrencoder.training.trainer import Trainer
    dataset = _toy_dataset()
    trainer = Trainer(V2_CONFIG, dataset, device="cpu")
    trainer.fit()   # must complete without error


def test_trainer_v2_phase1_use_transformer_false():
    from prostrencoder.training.trainer import Trainer
    dataset = _toy_dataset()
    trainer = Trainer(V2_CONFIG, dataset, device="cpu")
    # Before fit(), still in Phase 1 setup
    assert trainer.encoder.use_transformer is False


def test_trainer_v2_phase2_use_transformer_true():
    from prostrencoder.training.trainer import Trainer
    dataset = _toy_dataset()
    # warmup_phase_fraction=0 → Phase 2 immediately
    cfg = {**V2_CONFIG}
    cfg["training"] = {**V2_CONFIG["training"], "warmup_phase_fraction": 0.0}
    trainer = Trainer(cfg, dataset, device="cpu")
    trainer._transition_to_phase2()
    assert trainer.encoder.use_transformer is True


def test_trainer_v2_two_param_groups_in_phase2():
    from prostrencoder.training.trainer import Trainer
    dataset = _toy_dataset()
    cfg = {**V2_CONFIG}
    cfg["training"] = {**V2_CONFIG["training"], "warmup_phase_fraction": 0.0}
    trainer = Trainer(cfg, dataset, device="cpu")
    trainer._transition_to_phase2()
    assert len(trainer.optimizer.param_groups) == 2
    lr0 = trainer.optimizer.param_groups[0]["lr"]
    lr1 = trainer.optimizer.param_groups[1]["lr"]
    assert abs(lr0 / lr1 - 0.1) < 1e-6, "Group 0 must be 0.1x group 1 LR"


def test_trainer_v2_loss_decreases():
    from prostrencoder.training.trainer import Trainer
    torch.manual_seed(42)
    dataset = _toy_dataset(n_proteins=10)
    cfg = {**V2_CONFIG}
    cfg["training"] = {**V2_CONFIG["training"], "max_epochs": 4,
                       "warmup_phase_fraction": 0.0, "inverse_fold_prob": 0.0}
    trainer = Trainer(cfg, dataset, device="cpu")
    losses = []
    for epoch in range(1, 5):
        losses.append(trainer.train_epoch(epoch))
    # Loss should trend downward over 4 epochs on tiny dataset
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/training/test_trainer_v2.py::test_trainer_v2_two_epochs -v 2>&1 | tail -8
```
Expected: `TypeError` or `KeyError` from missing v2 config keys.

- [ ] **Step 3: Replace `prostrencoder/training/trainer.py`**

```python
# prostrencoder/training/trainer.py
import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from prostrencoder.models.encoder import ProStrEncoder
from prostrencoder.models.gnn import ResidueHead
from prostrencoder.training.objectives import (
    apply_masking, MaskedResidueLoss, InverseFoldingLoss, sample_batch_mode,
)


class Trainer:
    """
    Two-phase pre-training loop for ProStrEncoder v2.

    Phase 1 (warmup_phase_fraction of total steps):
      - GVP + WalkEncoder + bridge + masked_head in optimizer (single group)
      - encoder.use_transformer = False → bridge-only forward, no attention
      - Objective: masked residue prediction (Mode A) only

    Phase 2 (remaining steps):
      - encoder.use_transformer = True → full transformer forward
      - Optimizer group 0: GVP + WalkEncoder at lr * gvp_lr_multiplier
      - Optimizer group 1: transformer layers + heads at lr
      - Alternating batches: Mode A (1 - inverse_fold_prob) or Mode B (inverse_fold_prob)

    Precision:
      - precision=fp32 → no autocast (default, CPU-safe)
      - precision=bf16 → torch.amp.autocast BF16 (CUDA only, no GradScaler needed)
    """

    def __init__(self, config: dict, train_dataset, val_dataset=None,
                 device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        train_cfg = config["training"]

        self.encoder = ProStrEncoder(config["model"]).to(self.device)

        hidden_dim = self.encoder.hidden_dim
        self.masked_head   = ResidueHead(hidden_dim).to(self.device)
        self.inv_fold_head = ResidueHead(hidden_dim).to(self.device)

        self.masked_loss_fn   = MaskedResidueLoss()
        self.inv_fold_loss_fn = InverseFoldingLoss()

        self.mask_rate        = train_cfg["mask_rate"]
        self.grad_clip        = train_cfg["grad_clip"]
        self.inv_fold_prob    = train_cfg.get("inverse_fold_prob", 0.3)
        self.inv_fold_weight  = train_cfg.get("inverse_fold_weight", 1.0)
        self.precision        = train_cfg.get("precision", "fp32")
        self.log_every        = train_cfg["log_every"]
        self.save_every_steps = train_cfg.get("save_every_steps", 1000)

        num_workers = train_cfg.get("num_workers", 4)
        pin_memory  = (self.device.type == "cuda")

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg["batch_size"],
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        ) if val_dataset is not None else None

        total_steps = train_cfg["max_epochs"] * len(self.train_loader)
        warmup_steps = max(1, int(total_steps * train_cfg.get("warmup_phase_fraction", 0.05)))
        self._warmup_steps   = warmup_steps
        self._total_steps    = total_steps
        self._gvp_lr_mult    = train_cfg.get("gvp_lr_multiplier", 0.1)
        self._peak_lr        = train_cfg["lr"]
        self._weight_decay   = train_cfg["weight_decay"]
        self._step           = 0
        self._phase          = 1

        # Phase 1 optimizer: GVP + WalkEncoder + bridge + masked_head + out_proj
        self._build_phase1_optimizer()

        # Cosine LR schedule runs across both phases without reset
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)
        self.ckpt_dir = train_cfg["checkpoint_dir"]

    # ── Optimizer builders ─────────────────────────────────────────────────────

    def _phase1_params(self):
        """All params active during Phase 1."""
        params = (list(self.encoder.walk_sampler.parameters()) +
                  list(self.encoder.walk_encoder.parameters()) +
                  list(self.encoder.node_in.parameters()) +
                  list(self.encoder.edge_in.parameters()) +
                  list(self.encoder.layers.parameters()))
        if self.encoder.transformer is not None:
            params += list(self.encoder.transformer.bridge.parameters())
        params += (list(self.encoder.out_proj.parameters()) +
                   list(self.masked_head.parameters()))
        return params

    def _build_phase1_optimizer(self):
        self.params = self._phase1_params()
        self.optimizer = AdamW(self.params, lr=self._peak_lr,
                               weight_decay=self._weight_decay)

    def _transition_to_phase2(self):
        """Switch to Phase 2: activate transformer, add second LR group."""
        if self.encoder.transformer is None:
            # No transformer configured — stay as single group
            return
        self.encoder.use_transformer = True
        self._phase = 2

        # Group 0: GVP + WalkEncoder + bridge (conservative LR)
        gvp_params = (list(self.encoder.walk_sampler.parameters()) +
                      list(self.encoder.walk_encoder.parameters()) +
                      list(self.encoder.node_in.parameters()) +
                      list(self.encoder.edge_in.parameters()) +
                      list(self.encoder.layers.parameters()) +
                      list(self.encoder.transformer.bridge.parameters()))

        # Group 1: transformer layers + norm + both heads + out_proj (full LR)
        tx_params = (list(self.encoder.transformer.layers.parameters()) +
                     list(self.encoder.transformer.norm.parameters()) +
                     list(self.encoder.out_proj.parameters()) +
                     list(self.masked_head.parameters()) +
                     list(self.inv_fold_head.parameters()))

        current_lr = self.scheduler.get_last_lr()[0] if self._step > 0 else self._peak_lr
        self.optimizer = AdamW([
            {"params": gvp_params, "lr": current_lr * self._gvp_lr_mult},
            {"params": tx_params,  "lr": current_lr},
        ], weight_decay=self._weight_decay)

    # ── Forward + loss ──────────────────────────────────────────────────────────

    @property
    def _use_amp(self) -> bool:
        return self.precision == "bf16" and self.device.type == "cuda"

    def _forward_loss(self, batch):
        batch = batch.to(self.device)
        targets = batch.seq_idx.clone()

        # In Phase 1 always use Mode A; in Phase 2 alternate
        if self._phase == 1:
            mode = "A"
        else:
            mode = sample_batch_mode(self.inv_fold_prob)

        if mode == "A":
            masked_scalar, masked_seq_idx, mask = apply_masking(
                batch.seq_idx, batch.x_scalar, mask_rate=self.mask_rate
            )
            batch.x_scalar = masked_scalar
            batch.seq_idx  = masked_seq_idx

        amp_dtype = torch.bfloat16 if self._use_amp else torch.float32
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self._use_amp):
            _, hidden = self.encoder(batch, return_hidden=True)
            if mode == "A":
                logits = self.masked_head(hidden)
                loss   = self.masked_loss_fn(logits, targets, mask)
            else:
                logits  = self.inv_fold_head(hidden)
                weights = getattr(batch, "plddt", None)
                loss    = self.inv_fold_loss_fn(logits, targets, weights=weights)
                loss    = loss * self.inv_fold_weight

        return loss, mode

    # ── Training loop ──────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> float:
        self.encoder.train()
        self.masked_head.train()
        self.inv_fold_head.train()
        total_loss = 0.0
        t0 = time.time()

        for step_in_epoch, batch in enumerate(self.train_loader):
            # Phase transition check
            if self._phase == 1 and self._step >= self._warmup_steps:
                self._transition_to_phase2()

            self.optimizer.zero_grad()
            loss, mode = self._forward_loss(batch)
            loss.backward()

            all_params = []
            for pg in self.optimizer.param_groups:
                all_params += pg["params"]
            nn.utils.clip_grad_norm_(all_params, self.grad_clip)

            self.optimizer.step()
            self.scheduler.step()
            self._step += 1
            total_loss += loss.item()

            if (step_in_epoch + 1) % self.log_every == 0:
                lrs = [pg["lr"] for pg in self.optimizer.param_groups]
                record = {
                    "step": self._step, "epoch": epoch, "mode": mode,
                    "loss": round(loss.item(), 6),
                    "perplexity": round(math.exp(min(loss.item(), 20)), 4),
                    "lr": lrs[-1],
                    "phase": self._phase,
                }
                print(json.dumps(record))

            if self._step % self.save_every_steps == 0:
                self.save_checkpoint(epoch, float("nan"))

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def val_epoch(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.encoder.eval()
        self.masked_head.eval()
        total = 0.0
        for batch in self.val_loader:
            loss, _ = self._forward_loss(batch)
            total += loss.item()
        return total / len(self.val_loader)

    def save_checkpoint(self, epoch: int, val_loss: float):
        tag = f"step{self._step:07d}" if math.isnan(val_loss) else f"epoch{epoch:03d}_val{val_loss:.4f}"
        path = os.path.join(self.ckpt_dir, f"ckpt_{tag}.pt")
        torch.save({
            "epoch": epoch,
            "step": self._step,
            "phase": self._phase,
            "encoder": self.encoder.state_dict(),
            "masked_head": self.masked_head.state_dict(),
            "inv_fold_head": self.inv_fold_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
        print(json.dumps({"event": "checkpoint", "path": path}))

    def fit(self):
        best_val = math.inf
        for epoch in range(1, self.config["training"]["max_epochs"] + 1):
            train_loss = self.train_epoch(epoch)
            val_loss   = self.val_epoch()
            print(json.dumps({"event": "epoch", "epoch": epoch,
                               "train_loss": round(train_loss, 6),
                               "val_loss": round(val_loss, 6) if not math.isnan(val_loss) else None,
                               "phase": self._phase}))
            if val_loss < best_val:
                best_val = val_loss
                self.save_checkpoint(epoch, val_loss)
```

- [ ] **Step 4: Run trainer tests**

```bash
python -m pytest tests/training/test_trainer_v2.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Run existing test suite to verify backward compat**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -15
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add prostrencoder/training/trainer.py tests/training/test_trainer_v2.py
git commit -m "feat: v2 Trainer — two-phase schedule, layer-wise LR, alternating batch, BF16, JSON logging"
```

---

## Group C — Multi-GPU Train Script

---

### Task 6: Device Strategy — single_gpu / DDP / FSDP

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Write validation test**

Create `tests/test_train_script.py`:

```python
# tests/test_train_script.py
import os
import subprocess
import sys
import pytest


def _run(args):
    result = subprocess.run(
        [sys.executable, "scripts/train.py"] + args,
        capture_output=True, text=True,
    )
    return result


def test_missing_train_dir_raises():
    """Script must error clearly when train_dir is not set."""
    r = _run(["--config", "configs/default.yaml"])
    assert r.returncode != 0
    assert "train_dir" in r.stderr or "train_dir" in r.stdout


def test_single_gpu_with_local_rank_raises():
    """device_strategy=single_gpu + LOCAL_RANK set must raise clear error."""
    env = {**os.environ, "LOCAL_RANK": "0"}
    result = subprocess.run(
        [sys.executable, "scripts/train.py",
         "--config", "configs/default.yaml",
         "--train_dir", "data/toy/train"],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode != 0
    assert "torchrun" in result.stderr.lower() or "single_gpu" in result.stderr.lower()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_train_script.py::test_single_gpu_with_local_rank_raises -v 2>&1 | tail -5
```
Expected: test fails because the script doesn't yet check LOCAL_RANK.

- [ ] **Step 3: Replace `scripts/train.py`**

```python
# scripts/train.py
"""
Pre-train ProStrEncoder v2.

Device strategy is set in YAML (training.device_strategy):
  single_gpu  →  python scripts/train.py --config configs/medium.yaml
  ddp         →  torchrun --nproc_per_node=4 scripts/train.py --config configs/large.yaml
  fsdp        →  torchrun --nproc_per_node=4 scripts/train.py --config configs/xl.yaml
"""
import argparse
import json
import os
import sys

import torch
import torch.distributed as dist
import yaml

from prostrencoder.data.dataset import ProteinDataset
from prostrencoder.training.trainer import Trainer


def _setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def _wrap_ddp(encoder, masked_head, inv_fold_head, local_rank):
    from torch.nn.parallel import DistributedDataParallel as DDP
    encoder      = DDP(encoder,      device_ids=[local_rank])
    masked_head  = DDP(masked_head,  device_ids=[local_rank])
    inv_fold_head = DDP(inv_fold_head, device_ids=[local_rank])
    return encoder, masked_head, inv_fold_head


def _wrap_fsdp(encoder, masked_head, inv_fold_head):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    import torch
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    encoder      = FSDP(encoder,      mixed_precision=bf16_policy)
    masked_head  = FSDP(masked_head,  mixed_precision=bf16_policy)
    inv_fold_head = FSDP(inv_fold_head, mixed_precision=bf16_policy)
    return encoder, masked_head, inv_fold_head


def _save_fsdp_checkpoint(trainer, epoch, val_loss):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(trainer.encoder, StateDictType.FULL_STATE_DICT, cfg):
        encoder_state = trainer.encoder.state_dict()
    with FSDP.state_dict_type(trainer.masked_head, StateDictType.FULL_STATE_DICT, cfg):
        masked_state = trainer.masked_head.state_dict()
    with FSDP.state_dict_type(trainer.inv_fold_head, StateDictType.FULL_STATE_DICT, cfg):
        inv_fold_state = trainer.inv_fold_head.state_dict()
    if dist.get_rank() == 0:
        import math
        tag = (f"step{trainer._step:07d}" if math.isnan(val_loss)
               else f"epoch{epoch:03d}_val{val_loss:.4f}")
        path = os.path.join(trainer.ckpt_dir, f"ckpt_{tag}.pt")
        torch.save({
            "epoch": epoch,
            "step": trainer._step,
            "phase": trainer._phase,
            "encoder": encoder_state,
            "masked_head": masked_state,
            "inv_fold_head": inv_fold_state,
            "config": trainer.config,
        }, path)
        print(json.dumps({"event": "fsdp_checkpoint", "path": path}))


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train ProStrEncoder v2."
    )
    parser.add_argument("--config",    default="configs/default.yaml")
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir",   default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg    = config.get("training", {})
    data_cfg     = config.get("data", {})
    strategy     = train_cfg.get("device_strategy", "single_gpu")
    has_local_rank = "LOCAL_RANK" in os.environ

    # Validate launch method vs config
    if strategy == "single_gpu" and has_local_rank:
        print(
            "ERROR: device_strategy=single_gpu but LOCAL_RANK is set.\n"
            "       Do not use torchrun for single-GPU training.\n"
            "       Run: python scripts/train.py --config ...",
            file=sys.stderr,
        )
        sys.exit(1)
    if strategy in ("ddp", "fsdp") and not has_local_rank:
        print(
            f"ERROR: device_strategy={strategy} requires torchrun.\n"
            f"       Run: torchrun --nproc_per_node=N scripts/train.py --config ...",
            file=sys.stderr,
        )
        sys.exit(1)

    # Data paths
    train_dir = args.train_dir or data_cfg.get("train_dir")
    val_dir   = args.val_dir   or data_cfg.get("val_dir")
    if not train_dir:
        print("ERROR: train_dir not set. Provide --train_dir or config.data.train_dir",
              file=sys.stderr)
        sys.exit(1)

    # Distributed setup
    local_rank = 0
    world_size = 1
    if strategy in ("ddp", "fsdp"):
        local_rank, world_size = _setup_ddp()

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    is_main = (local_rank == 0)

    train_ds = ProteinDataset(train_dir)
    val_ds   = ProteinDataset(val_dir) if val_dir else None

    if is_main:
        print(json.dumps({"event": "start", "strategy": strategy,
                          "world_size": world_size,
                          "train_size": len(train_ds),
                          "val_size": len(val_ds) if val_ds else 0}))

    trainer = Trainer(config, train_ds, val_ds, device=device)

    # Wrap with DDP / FSDP
    if strategy == "ddp":
        trainer.encoder, trainer.masked_head, trainer.inv_fold_head = _wrap_ddp(
            trainer.encoder, trainer.masked_head, trainer.inv_fold_head, local_rank
        )
    elif strategy == "fsdp":
        trainer.encoder, trainer.masked_head, trainer.inv_fold_head = _wrap_fsdp(
            trainer.encoder, trainer.masked_head, trainer.inv_fold_head
        )
        # Override save_checkpoint for FSDP
        import math
        trainer.save_checkpoint = lambda epoch, val_loss: _save_fsdp_checkpoint(
            trainer, epoch, val_loss
        )

    # DDP: use DistributedSampler so each rank sees different data
    if strategy == "ddp":
        from torch.utils.data import DistributedSampler
        from torch_geometric.loader import DataLoader
        sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                     rank=local_rank, shuffle=True)
        trainer.train_loader = DataLoader(
            train_ds,
            batch_size=config["training"]["batch_size"],
            sampler=sampler,
            num_workers=config["training"].get("num_workers", 4),
            pin_memory=True,
        )

    trainer.fit()

    if strategy in ("ddp", "fsdp"):
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run validation tests**

```bash
python -m pytest tests/test_train_script.py -v
```
Expected: all pass.

- [ ] **Step 5: Smoke test single GPU**

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --train_dir data/toy/train \
  --val_dir data/toy/val 2>&1 | head -10
```
Expected: JSON log lines printed, no errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/train.py tests/test_train_script.py
git commit -m "feat: train.py — device_strategy single_gpu/ddp/fsdp, torchrun validation, FSDP checkpoint"
```

---

## Group D — Data Pipeline (Independent)

---

### Task 7: Parallel Preprocessing

**Files:**
- Modify: `scripts/preprocess.py`

- [ ] **Step 1: Replace the sequential loop in `main()` in `scripts/preprocess.py`**

The function `preprocess_one` is unchanged. Only the `main()` function changes.
Replace everything from `os.makedirs(out_dir, ...)` to the end of `main()`:

```python
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess PDB/mmCIF files into PyG .pt graph objects."
    )
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--pdb_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--k",       type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    pdb_dir  = args.pdb_dir or data_cfg.get("pdb_dir", "data/pdb")
    out_dir  = args.out_dir or data_cfg.get("processed_dir", "data/processed")
    k        = args.k if args.k is not None else data_cfg.get("k_neighbors", 30)
    n_workers = data_cfg.get("preprocess_workers", 1)

    print(f"pdb_dir  : {pdb_dir}")
    print(f"out_dir  : {out_dir}")
    print(f"k-NN     : {k}")
    print(f"workers  : {n_workers}")

    os.makedirs(out_dir, exist_ok=True)
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith((".pdb", ".cif"))]
    pending   = [f for f in pdb_files
                 if not os.path.exists(os.path.join(out_dir,
                                                     os.path.splitext(f)[0] + ".pt"))]
    print(f"Files pending: {len(pending)}/{len(pdb_files)}")

    if not pending:
        print("Nothing to do.")
        return

    def _worker(fname):
        stem     = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, stem + ".pt")
        try:
            data = preprocess_one(os.path.join(pdb_dir, fname), k)
            torch.save(data, out_path)
            return None
        except Exception as e:
            return f"{fname}: {e}"

    if n_workers <= 1:
        errors = [_worker(f) for f in tqdm(pending, desc="Preprocessing")]
    else:
        from multiprocessing import Pool
        with Pool(processes=n_workers) as pool:
            errors = list(tqdm(pool.imap(_worker, pending),
                               total=len(pending), desc="Preprocessing"))

    skipped = [e for e in errors if e is not None]
    for e in skipped:
        print(f"Skipped: {e}")
    print(f"Done. Skipped {len(skipped)}/{len(pending)} files.")
```

- [ ] **Step 2: Verify it still runs correctly on toy data**

```bash
python scripts/preprocess.py --config configs/default.yaml \
  --pdb_dir data/pdb --out_dir /tmp/test_preprocess_parallel --k 30 2>&1
```
Expected: `Files pending: 0/0` (no PDB files in data/pdb) or processes correctly with any PDB files present.

- [ ] **Step 3: Commit**

```bash
git add scripts/preprocess.py
git commit -m "feat: parallel preprocessing with multiprocessing.Pool (preprocess_workers config key)"
```

---

### Task 8: PDB Download and Quality Filtering

**Files:**
- Create: `scripts/download_pdb.py`

- [ ] **Step 1: Create `scripts/download_pdb.py`**

```python
# scripts/download_pdb.py
"""
Download PDB structures from RCSB and apply per-chain quality filters.

Filters (all configurable via YAML):
  - Resolution ≤ resolution_cutoff Å (X-ray; cryo-EM always accepted)
  - ≥ min_chain_coverage fraction of residues have all 4 backbone atoms
  - Chain length within [min_len, max_len]

After filtering, runs MMseqs2 at cluster_seqid identity to split train/val
and prevent homolog leakage.

Usage:
    python scripts/download_pdb.py --config configs/medium.yaml
    python scripts/download_pdb.py --config configs/medium.yaml --skip_download
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import yaml

from prostrencoder.data.parser import parse_structure


def _rsync_pdb(pdb_dir: str):
    """Mirror RCSB PDB using rsync (requires rsync installed)."""
    os.makedirs(pdb_dir, exist_ok=True)
    cmd = [
        "rsync", "-rlpt", "-v", "-z",
        "--delete",
        "--port=33444",
        "rsync.rcsb.org::ftp_data/structures/divided/mmCIF/",
        pdb_dir,
    ]
    print(f"Syncing PDB from RCSB → {pdb_dir}")
    print("(This can take several hours for the full PDB)")
    subprocess.run(cmd, check=True)


def _chain_passes_filters(pdb_path: str, resolution_cutoff: float,
                          min_chain_coverage: float, min_len: int,
                          max_len: int) -> bool:
    """Return True if the structure passes all quality filters."""
    try:
        parsed = parse_structure(pdb_path)
    except Exception:
        return False

    n = len(parsed["seq_idx"])
    if n < min_len or n > max_len:
        return False

    bb = parsed["backbone_coords"]   # (N, 4, 3)
    n_complete = int(np.sum(~np.any(np.isnan(bb.reshape(n, -1)), axis=1)))
    if n_complete / n < min_chain_coverage:
        return False

    # Resolution check: stored in REMARK 2 of PDB; BioPython parser puts it
    # in structure.header["resolution"]. If unavailable (cryo-EM), pass.
    try:
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("X", pdb_path)
        resolution = structure.header.get("resolution")
        if resolution is not None and float(resolution) > resolution_cutoff:
            return False
    except Exception:
        pass   # If we can't read resolution, don't filter

    return True


def _write_fasta(chain_paths: list, fasta_path: str):
    """Write a FASTA file from chain paths for MMseqs2 clustering."""
    from prostrencoder.data.parser import parse_structure
    AA_1LETTER = "ARNDCQEGHILKMFPSTWYV"
    with open(fasta_path, "w") as f:
        for path in chain_paths:
            try:
                parsed = parse_structure(path)
                seq = "".join(AA_1LETTER[i] if i < 20 else "X"
                              for i in parsed["seq_idx"])
                f.write(f">{path}\n{seq}\n")
            except Exception:
                pass


def _run_mmseqs2(fasta_path: str, cluster_seqid: float, out_dir: str) -> dict:
    """
    Cluster sequences with MMseqs2. Returns dict mapping repr → [members].
    Requires mmseqs2 installed (conda install -c bioconda mmseqs2).
    """
    db     = os.path.join(out_dir, "mmseqs_db")
    result = os.path.join(out_dir, "mmseqs_result")
    tsv    = os.path.join(out_dir, "mmseqs_cluster.tsv")
    tmp    = os.path.join(out_dir, "mmseqs_tmp")
    os.makedirs(tmp, exist_ok=True)

    subprocess.run(["mmseqs", "createdb", fasta_path, db], check=True)
    subprocess.run(["mmseqs", "cluster", db, result, tmp,
                    "--min-seq-id", str(cluster_seqid),
                    "-c", "0.8", "--cov-mode", "0"], check=True)
    subprocess.run(["mmseqs", "createtsv", db, db, result, tsv], check=True)

    clusters = {}
    with open(tsv) as f:
        for line in f:
            repr_id, member = line.strip().split("\t")
            clusters.setdefault(repr_id, []).append(member)
    return clusters


def _split_clusters(clusters: dict, val_frac: float = 0.05):
    """Split cluster representatives into train and val sets."""
    reprs = sorted(clusters.keys())
    n_val = max(1, int(len(reprs) * val_frac))
    import random; random.shuffle(reprs)
    val_reprs   = set(reprs[:n_val])
    train_reprs = set(reprs[n_val:])

    train_paths, val_paths = [], []
    for r, members in clusters.items():
        if r in val_reprs:
            val_paths.extend(members)
        else:
            train_paths.extend(members)
    return train_paths, val_paths


def main():
    parser = argparse.ArgumentParser(
        description="Download PDB and apply quality filters."
    )
    parser.add_argument("--config",        default="configs/medium.yaml")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip rsync; use existing pdb_dir files")
    parser.add_argument("--val_frac",      type=float, default=0.05,
                        help="Fraction of clusters reserved for validation")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    pdb_dir  = data_cfg["pdb_dir"]
    out_base = data_cfg["processed_dir"]
    res_cut  = data_cfg.get("resolution_cutoff", 3.5)
    min_cov  = data_cfg.get("min_chain_coverage", 0.9)
    min_len  = data_cfg.get("min_len", 40)
    max_len  = data_cfg.get("max_len", 1024)
    seqid    = data_cfg.get("cluster_seqid", 0.30)

    if not args.skip_download:
        _rsync_pdb(pdb_dir)

    # Collect all structure files
    pdb_files = []
    for root, _, files in os.walk(pdb_dir):
        for f in files:
            if f.endswith((".cif", ".cif.gz", ".pdb", ".pdb.gz")):
                pdb_files.append(os.path.join(root, f))
    print(f"Found {len(pdb_files)} structure files")

    # Apply quality filters
    print("Applying quality filters …")
    passing = [p for p in pdb_files
               if _chain_passes_filters(p, res_cut, min_cov, min_len, max_len)]
    print(f"Passing filters: {len(passing)}/{len(pdb_files)}")

    # Write FASTA and cluster
    mmseqs_dir = os.path.join(out_base, "mmseqs")
    os.makedirs(mmseqs_dir, exist_ok=True)
    fasta_path = os.path.join(mmseqs_dir, "chains.fasta")
    _write_fasta(passing, fasta_path)

    print(f"Running MMseqs2 at {seqid*100:.0f}% identity …")
    clusters = _run_mmseqs2(fasta_path, seqid, mmseqs_dir)
    print(f"Clusters: {len(clusters)}")

    train_paths, val_paths = _split_clusters(clusters, val_frac=args.val_frac)
    print(f"Train chains: {len(train_paths)}, Val chains: {len(val_paths)}")

    # Write split manifest files
    for split, paths in [("train", train_paths), ("val", val_paths)]:
        manifest = os.path.join(out_base, f"{split}_manifest.txt")
        with open(manifest, "w") as f:
            f.write("\n".join(paths))
        print(f"Written: {manifest}")

    print("\nNext step: run scripts/preprocess.py for each split")
    print(f"  python scripts/preprocess.py --config {args.config} "
          f"--pdb_dir {pdb_dir} --out_dir {out_base}/train")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script is importable**

```bash
python -c "import scripts.download_pdb" 2>&1 || python scripts/download_pdb.py --help
```
Expected: help text printed, no import errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/download_pdb.py
git commit -m "feat: download_pdb.py — RCSB rsync, resolution/coverage/length filters, MMseqs2 train/val split"
```

---

## Group E — Evaluation

---

### Task 9: Inverse Folding Evaluation Script

**Files:**
- Create: `scripts/eval_inverse_folding.py`

- [ ] **Step 1: Create the evaluation script**

```python
# scripts/eval_inverse_folding.py
"""
Evaluate the inverse folding head's native sequence recovery rate.

For each held-out structure: given the full 3D backbone, the inverse folding
head predicts a sequence. Native sequence recovery = fraction of positions
where the predicted AA matches the true AA.

Usage:
    python scripts/eval_inverse_folding.py \\
        --checkpoint checkpoints/medium/ckpt_epoch010_val1.2345.pt \\
        --test_dir   data/processed/val \\
        --config     configs/medium.yaml \\
        --device     cuda
"""
import argparse
import json
import math

import torch
import yaml
from torch_geometric.loader import DataLoader

from prostrencoder.data.dataset import ProteinDataset
from prostrencoder.models.encoder import ProStrEncoder
from prostrencoder.models.gnn import ResidueHead


@torch.no_grad()
def evaluate(encoder: ProStrEncoder, inv_fold_head: ResidueHead,
             loader: DataLoader, device: torch.device) -> dict:
    encoder.eval()
    inv_fold_head.eval()

    total_residues  = 0
    correct_residues = 0
    total_ce_loss   = 0.0
    n_batches       = 0

    import torch.nn.functional as F

    for batch in loader:
        batch = batch.to(device)
        targets = batch.seq_idx.clone()

        # Inverse folding: full structure in, no masking
        _, hidden = encoder(batch, return_hidden=True)
        logits    = inv_fold_head(hidden)              # (N, 21)

        # Per-residue accuracy
        preds   = logits.argmax(dim=-1)                # (N,)
        correct = (preds == targets).sum().item()
        correct_residues += correct
        total_residues   += targets.shape[0]

        # Per-residue cross-entropy (= negative log-likelihood)
        ce = F.cross_entropy(logits, targets, reduction="mean")
        total_ce_loss += ce.item()
        n_batches += 1

    recovery    = correct_residues / max(total_residues, 1)
    mean_ce     = total_ce_loss / max(n_batches, 1)
    perplexity  = math.exp(min(mean_ce, 20))

    return {
        "sequence_recovery": round(recovery, 4),
        "mean_ce_loss":      round(mean_ce, 6),
        "perplexity":        round(perplexity, 4),
        "total_residues":    total_residues,
        "correct_residues":  correct_residues,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate inverse folding sequence recovery."
    )
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--test_dir",    required=True,
                        help="Directory of processed .pt structure files")
    parser.add_argument("--config",      default="configs/medium.yaml")
    parser.add_argument("--batch_size",  type=int, default=None,
                        help="Override config batch_size")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device      = torch.device(args.device)
    model_cfg   = config["model"]
    batch_size  = args.batch_size or config["training"]["batch_size"]

    # Build model
    encoder = ProStrEncoder(model_cfg).to(device)
    encoder.use_transformer = True   # Phase 2 full forward

    hidden_dim    = encoder.hidden_dim
    inv_fold_head = ResidueHead(hidden_dim).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    inv_fold_head.load_state_dict(ckpt["inv_fold_head"])
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  trained for {ckpt.get('step', '?')} steps, phase {ckpt.get('phase', '?')}")

    dataset = ProteinDataset(args.test_dir)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0)
    print(f"Evaluating on {len(dataset)} structures from {args.test_dir}")

    results = evaluate(encoder, inv_fold_head, loader, device)
    print(json.dumps(results, indent=2))

    recovery_pct = results["sequence_recovery"] * 100
    if recovery_pct >= 35:
        print(f"\n✓ Recovery {recovery_pct:.1f}% meets target (≥35%)")
    else:
        print(f"\n✗ Recovery {recovery_pct:.1f}% below target (35%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script imports cleanly**

```bash
python -c "import scripts.eval_inverse_folding" 2>&1 || python scripts/eval_inverse_folding.py --help
```
Expected: help text printed, no errors.

- [ ] **Step 3: Smoke test on toy data with the existing Small checkpoint**

```bash
# First generate a checkpoint via a quick training run
python -c "
import torch, yaml
from prostrencoder.data.dataset import ProteinDataset
from prostrencoder.training.trainer import Trainer

with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['training']['max_epochs'] = 1
cfg['training']['num_workers'] = 0

trainer = Trainer(cfg, ProteinDataset('data/toy/train'),
                  ProteinDataset('data/toy/val'), device='cpu')
trainer.fit()
" 2>&1 | tail -3

# Identify the checkpoint and run eval
ls checkpoints/*.pt | head -1
python scripts/eval_inverse_folding.py \
  --checkpoint $(ls checkpoints/*.pt | head -1) \
  --test_dir data/toy/val \
  --config configs/default.yaml \
  --device cpu 2>&1
```
Expected: JSON output with `sequence_recovery` between 0.0 and 1.0.

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_inverse_folding.py
git commit -m "feat: eval_inverse_folding.py — sequence recovery rate + perplexity evaluation"
```

---

## Final Validation

- [ ] **Run the full test suite**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```
Expected: all tests pass.

- [ ] **End-to-end smoke test with Medium config (tx_layers=2 for speed)**

```python
# Run in Python
import torch, yaml
from prostrencoder.data.dataset import ProteinDataset
from prostrencoder.training.trainer import Trainer

with open('configs/medium.yaml') as f:
    cfg = yaml.safe_load(f)

# Override for quick test
cfg['model']['tx_layers'] = 2
cfg['model']['num_layers'] = 2
cfg['model']['hidden_scalar'] = 64
cfg['model']['hidden_vector'] = 8
cfg['model']['node_scalar_in'] = 155   # 27 + 128
cfg['model']['tx_d_model'] = 64
cfg['model']['tx_num_heads'] = 4
cfg['training']['max_epochs'] = 2
cfg['training']['num_workers'] = 0
cfg['training']['precision'] = 'fp32'

trainer = Trainer(cfg, ProteinDataset('data/toy/train'),
                  ProteinDataset('data/toy/val'), device='cpu')
trainer.fit()
print('End-to-end OK')
```
Expected: `End-to-end OK`

- [ ] **Final commit**

```bash
git add -A
git commit -m "feat: ProStrEncoder v2 — full implementation complete"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] GlobalTransformer with Flash Attention 2 + RoPE + sliding window → Task 1
- [x] GVP → bridge → transformer integration + `_transformer_forward` → Task 2
- [x] Config tiers (medium/large/xl) + backward compat → Task 3
- [x] Inverse folding loss + alternating batch + `sample_batch_mode` → Task 4
- [x] Two-phase trainer, layer-wise LR, BF16, gradient checkpointing → Task 5
- [x] Device strategy validation, DDP, FSDP, FSDP checkpoint → Task 6
- [x] Parallel preprocessing → Task 7
- [x] PDB download + quality filtering + MMseqs2 → Task 8
- [x] Inverse folding evaluation → Task 9
- [x] JSON structured logging → Task 5 (trainer._log via json.dumps)
- [x] W&B flag — config key exists in YAMLs; hookup deferred (no task needed until W&B is installed)
- [x] `ShardedProteinDataset` (WebDataset) — noted in config (dataset_format key); implementation deferred to AFDB augmentation phase, not in current plan scope

**Type consistency:**
- `encoder.hidden_dim` → used in Task 2 (definition) and Task 5 (trainer reads it)
- `trainer.masked_head` / `trainer.inv_fold_head` → defined in Task 5, read in Task 6 (DDP wrap) and Task 9 (checkpoint)
- `trainer._transition_to_phase2()` → defined in Task 5, tested in Task 5 tests
- `InverseFoldingLoss` / `sample_batch_mode` → defined in Task 4, imported in Task 5
- `GlobalTransformer.d_model` property → defined in Task 1, read by `encoder.hidden_dim` in Task 2
- `ckpt["inv_fold_head"]` → saved in Task 5 `save_checkpoint`, loaded in Task 9 `eval_inverse_folding`

**No placeholders found.**
