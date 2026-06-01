# CyberGFM Option C — Walk-Sequence Embedding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 16-dim RWSE landing-probability encoding in ProStrEncoder with a richer 64-dim walk-sequence embedding produced by a small CyberGFM-inspired transformer, while keeping all 48 existing tests green.

**Architecture:** For each residue *i*, a `WalkSampler` draws K=8 random walks of length L=20 on the protein k-NN graph and converts them to AA-type token sequences (vocab 0–21; 21 = CLS). A `WalkEncoder` (4-layer transformer, 64-dim) encodes each sequence, mean-pools the K CLS-token outputs, and returns a 64-dim embedding per residue. This embedding is concatenated with the existing 27-dim scalar features (AA one-hot + torsion) inside `ProStrEncoder.forward()`, yielding a 91-dim input to the GVP layers — replacing the old 43-dim input (27 + 16 RWSE). Both new modules are trained end-to-end with the existing masked residue prediction objective.

**Tech Stack:** PyTorch 2.x, PyTorch Geometric 2.4+, existing ProStrEncoder codebase.

---

## File Map

```
# New files
prostrencoder/models/walk_sampler.py   ← WalkSampler: edge_index + seq_idx → token walks
prostrencoder/models/walk_encoder.py   ← WalkEncoder: token walks → 64-dim embeddings
tests/models/test_walk_sampler.py
tests/models/test_walk_encoder.py

# Modified files
prostrencoder/models/encoder.py        ← add WalkSampler + WalkEncoder submodules
tests/models/test_encoder.py           ← update CONFIG + fake-batch x_scalar dim
scripts/preprocess.py                  ← drop compute_rwse; x_scalar → (N, 27)
tests/data/test_dataset.py             ← update fixture + shape assertion (43 → 27)
configs/default.yaml                   ← node_scalar_in 43→91; add walk params
```

**Feature dimension changes:**

| Field | Before | After | Why |
|---|---|---|---|
| `x_scalar` in `.pt` files | (N, 43) = 21+6+16 | (N, 27) = 21+6 | RWSE removed from preprocessing |
| walk embedding (online) | — | (N, 64) | WalkEncoder output |
| `node_s` entering GVP | (N, 43) | (N, 91) = 27+64 | concatenated inside `forward()` |
| `node_scalar_in` in config | 43 | 91 | drives GVP input projection |

---

### Task 1: WalkSampler

**Files:**
- Create: `prostrencoder/models/walk_sampler.py`
- Create: `tests/models/test_walk_sampler.py`

- [ ] **Step 1: Write the failing test**

`tests/models/test_walk_sampler.py`:

```python
import torch
import pytest
from prostrencoder.models.walk_sampler import WalkSampler


def chain_graph(n):
    """Bidirectional chain 0↔1↔2↔…↔(n-1)."""
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    return torch.tensor([src, dst], dtype=torch.long)


SAMPLER = WalkSampler(num_walks=4, walk_length=5)


def test_output_shape():
    n = 10
    edge_index = chain_graph(n)
    seq_idx = torch.randint(0, 21, (n,))
    out = SAMPLER(edge_index, seq_idx)
    # seq = [CLS, start_aa, step1_aa, ..., step5_aa] → length 7
    assert out.shape == (n, 4, 7)   # (N, num_walks, walk_length + 2)
    assert out.dtype == torch.long


def test_first_token_is_cls():
    """Every walk must start with the CLS token (index 21)."""
    n = 8
    edge_index = chain_graph(n)
    seq_idx = torch.arange(n) % 21
    out = SAMPLER(edge_index, seq_idx)
    assert torch.all(out[:, :, 0] == WalkSampler.CLS_TOKEN)


def test_second_token_is_starting_residue_aa():
    """Position 1 in every walk must be the AA type of the source residue."""
    n = 8
    edge_index = chain_graph(n)
    seq_idx = torch.arange(n) % 21
    out = SAMPLER(edge_index, seq_idx)
    for i in range(n):
        assert torch.all(out[i, :, 1] == seq_idx[i])


def test_token_values_in_valid_range():
    """All token values must be in 0–21 (0–20 AA, 21 CLS)."""
    n = 15
    edge_index = chain_graph(n)
    seq_idx = torch.randint(0, 21, (n,))
    out = SAMPLER(edge_index, seq_idx)
    assert torch.all(out >= 0)
    assert torch.all(out <= WalkSampler.CLS_TOKEN)


def test_isolated_node_stays_put():
    """An isolated node (degree 0) must stay at itself for all walk steps."""
    # chain 0-1-2, node 3 is isolated
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    seq_idx = torch.tensor([5, 6, 7, 8])
    out = SAMPLER(edge_index, seq_idx)
    # node 3: position 0 = CLS, positions 1..6 = seq_idx[3] = 8
    assert torch.all(out[3, :, 0] == WalkSampler.CLS_TOKEN)
    assert torch.all(out[3, :, 1:] == 8)


def test_multiple_walks_can_differ():
    """On a branching graph, K > 1 walks from the same node should not all be equal."""
    # star graph: node 0 → {1,2,3,4} bidirectionally
    src = [0, 0, 0, 0, 1, 2, 3, 4]
    dst = [1, 2, 3, 4, 0, 0, 0, 0]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    seq_idx = torch.arange(5) % 21
    sampler = WalkSampler(num_walks=16, walk_length=3)
    out = sampler(edge_index, seq_idx)
    walks_from_hub = out[0]   # (16, 5)
    # Not all 16 walks should be identical (star has 4 choices at each step)
    assert not torch.all(walks_from_hub == walks_from_hub[0])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/models/test_walk_sampler.py -v
```

Expected: `ImportError: cannot import name 'WalkSampler'`

- [ ] **Step 3: Write `prostrencoder/models/walk_sampler.py`**

```python
import torch
import torch.nn as nn


class WalkSampler(nn.Module):
    """
    Samples random walks on a protein residue graph and returns AA-type
    token sequences for each residue.

    Not trainable — no parameters. Subclasses nn.Module for device consistency.

    For each residue i, draws num_walks independent random walks of walk_length
    steps on the k-NN graph defined by edge_index. Each walk is converted to
    a sequence of AA-type tokens and prepended with a CLS token.

    Args:
        num_walks  : K — number of independent walks per residue.
        walk_length: L — number of steps per walk (excluding starting node).

    Output token layout (length L + 2):
        [CLS(21), aa(i), aa(step1), aa(step2), ..., aa(stepL)]
    """

    CLS_TOKEN: int = 21

    def __init__(self, num_walks: int = 8, walk_length: int = 20):
        super().__init__()
        self.num_walks = num_walks
        self.walk_length = walk_length

    def forward(self, edge_index: torch.Tensor,
                seq_idx: torch.Tensor) -> torch.Tensor:
        """
        edge_index : (2, E) int64
        seq_idx    : (N,) int64 — AA type per residue (0–20)

        Returns: (N, num_walks, walk_length + 2) int64
        """
        N = seq_idx.shape[0]
        device = seq_idx.device
        src, dst = edge_index[0], edge_index[1]

        # ── Build padded adjacency tensor ─────────────────────────────────────
        degree = torch.zeros(N, dtype=torch.long, device=device)
        degree.scatter_add_(0, src, torch.ones_like(src))
        max_deg = max(int(degree.max().item()), 1)

        # Sort edges by source node so we can compute within-group slot indices
        order = src.argsort(stable=True)
        s_src, s_dst = src[order], dst[order]

        # cum_deg[i] = total number of edges with src < i (= start of node i's block)
        cum_deg = torch.zeros(N + 1, dtype=torch.long, device=device)
        cum_deg[1:] = degree.cumsum(0)
        slot = torch.arange(s_src.shape[0], device=device) - cum_deg[s_src]

        adj = torch.full((N, max_deg), -1, dtype=torch.long, device=device)
        adj[s_src, slot] = s_dst

        # ── Vectorised walk sampling ──────────────────────────────────────────
        # current[i, k] = node currently occupied by walk k of residue i
        current = (torch.arange(N, device=device)
                   .unsqueeze(1)
                   .expand(N, self.num_walks))                # (N, K)
        walk_nodes = [current]

        for _ in range(self.walk_length):
            deg = degree[current].clamp(min=1)               # (N, K)
            rand_slot = (torch.rand(N, self.num_walks, device=device)
                         * deg.float()).long().clamp(0, max_deg - 1)
            nxt = adj[current, rand_slot]                    # (N, K)
            # Stay in place when no valid neighbour (-1)
            nxt = torch.where(nxt >= 0, nxt, current)
            current = nxt
            walk_nodes.append(current)

        walk_nodes = torch.stack(walk_nodes, dim=2)          # (N, K, L+1)

        # ── Convert node indices → AA token IDs ──────────────────────────────
        walk_tokens = seq_idx[walk_nodes]                    # (N, K, L+1)

        # Prepend CLS
        cls = torch.full((N, self.num_walks, 1), self.CLS_TOKEN,
                         dtype=torch.long, device=device)
        return torch.cat([cls, walk_tokens], dim=2)          # (N, K, L+2)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/models/test_walk_sampler.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add prostrencoder/models/walk_sampler.py tests/models/test_walk_sampler.py
git commit -m "feat: WalkSampler — random-walk token sequences for protein residue graphs"
```

---

### Task 2: WalkEncoder

**Files:**
- Create: `prostrencoder/models/walk_encoder.py`
- Create: `tests/models/test_walk_encoder.py`

- [ ] **Step 1: Write the failing test**

`tests/models/test_walk_encoder.py`:

```python
import torch
import pytest
from prostrencoder.models.walk_encoder import WalkEncoder


# Small encoder for fast tests
ENC = WalkEncoder(
    vocab_size=22, embed_dim=64, num_heads=4,
    num_layers=2, ff_dim=128, max_seq_len=24,
)


def test_output_shape():
    N, K, L = 10, 4, 7
    tokens = torch.randint(0, 22, (N, K, L))
    out = ENC(tokens)
    assert out.shape == (N, 64)
    assert out.dtype == torch.float32


def test_output_is_finite():
    tokens = torch.randint(0, 22, (8, 4, 7))
    out = ENC(tokens)
    assert torch.all(torch.isfinite(out))


def test_different_walks_change_output():
    """Changing one residue's walk tokens must change only that residue's embedding."""
    ENC.eval()
    tokens1 = torch.randint(0, 21, (6, 4, 7))
    tokens2 = tokens1.clone()
    tokens2[0, :, 3] = (tokens2[0, :, 3] + 5) % 21   # mutate residue 0
    with torch.no_grad():
        out1 = ENC(tokens1)
        out2 = ENC(tokens2)
    assert not torch.allclose(out1[0], out2[0])        # residue 0 changed
    torch.testing.assert_close(out1[1:], out2[1:])     # residues 1‥5 unchanged


def test_single_walk_valid():
    """K=1 must still produce correct (N, embed_dim) output."""
    enc = WalkEncoder(vocab_size=22, embed_dim=32, num_heads=2,
                      num_layers=1, ff_dim=64, max_seq_len=24)
    tokens = torch.randint(0, 22, (5, 1, 7))
    out = enc(tokens)
    assert out.shape == (5, 32)


def test_deterministic_in_eval_mode():
    """eval() + no dropout → identical outputs for same input."""
    ENC.eval()
    tokens = torch.randint(0, 22, (6, 4, 7))
    with torch.no_grad():
        out1 = ENC(tokens)
        out2 = ENC(tokens)
    torch.testing.assert_close(out1, out2)


def test_cls_position_matters():
    """Replacing the non-CLS tokens with random values should change output."""
    ENC.eval()
    N, K, L = 4, 4, 7
    tokens_a = torch.randint(0, 21, (N, K, L))
    tokens_a[:, :, 0] = 21   # CLS at position 0
    tokens_b = tokens_a.clone()
    tokens_b[:, :, 1:] = torch.randint(0, 21, (N, K, L - 1))
    with torch.no_grad():
        out_a = ENC(tokens_a)
        out_b = ENC(tokens_b)
    assert not torch.allclose(out_a, out_b)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/models/test_walk_encoder.py -v
```

Expected: `ImportError: cannot import name 'WalkEncoder'`

- [ ] **Step 3: Write `prostrencoder/models/walk_encoder.py`**

```python
import torch
import torch.nn as nn


class WalkEncoder(nn.Module):
    """
    Encodes random-walk token sequences into per-residue scalar embeddings.

    Inspired by CyberGFM's walk-tokenizer + BERT design (cybermonic/CyberGFM).

    For each residue, K walk sequences are encoded independently by a
    transformer encoder. The CLS-token hidden state from each walk is
    extracted and the K states are mean-pooled to produce one embedding
    per residue.

    The output is rotation-invariant because it depends only on AA-type
    token IDs (integers), never on 3D coordinates.

    Args:
        vocab_size  : number of distinct token IDs (default 22: 0-20 AA + 21 CLS).
        embed_dim   : transformer hidden dimension and output dimension.
        num_heads   : attention heads (embed_dim must be divisible by num_heads).
        num_layers  : number of transformer encoder layers.
        ff_dim      : feed-forward hidden dimension inside each transformer layer.
        max_seq_len : maximum sequence length (must be ≥ walk_length + 2).
        dropout     : dropout rate applied inside transformer layers.
    """

    def __init__(self, vocab_size: int = 22, embed_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 4,
                 ff_dim: int = 256, max_seq_len: int = 24,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed   = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # pre-norm: more stable for shallow transformers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, walk_tokens: torch.Tensor) -> torch.Tensor:
        """
        walk_tokens : (N, K, L) int64
            N = number of residues, K = num_walks, L = walk_length + 2
            Position 0 of each sequence must be the CLS token (index 21).

        Returns: (N, embed_dim) float32 — one embedding per residue.
        """
        N, K, L = walk_tokens.shape
        device = walk_tokens.device

        # Flatten (N, K) into a single batch dimension: (N*K, L)
        tokens = walk_tokens.reshape(N * K, L)

        # Token + positional embeddings
        pos = torch.arange(L, device=device).unsqueeze(0)   # (1, L)
        x = self.token_embed(tokens) + self.pos_embed(pos)  # (N*K, L, D)

        # Transformer encoding
        x = self.transformer(x)    # (N*K, L, D)
        x = self.norm(x)

        # Extract CLS hidden state (position 0)
        cls_out = x[:, 0, :]                   # (N*K, D)

        # Mean-pool K walks per residue
        cls_out = cls_out.reshape(N, K, -1)    # (N, K, D)
        return cls_out.mean(dim=1)             # (N, D)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/models/test_walk_encoder.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add prostrencoder/models/walk_encoder.py tests/models/test_walk_encoder.py
git commit -m "feat: WalkEncoder — 4-layer transformer walk-sequence embeddings (CyberGFM-inspired)"
```

---

### Task 3: Update ProStrEncoder

**Files:**
- Modify: `prostrencoder/models/encoder.py`
- Modify: `tests/models/test_encoder.py`

- [ ] **Step 1: Update the encoder test first (test-first on the new interface)**

Replace the entire content of `tests/models/test_encoder.py` with:

```python
import torch
import pytest
from torch_geometric.data import Data, Batch
from prostrencoder.models.encoder import ProStrEncoder


def make_fake_batch(n_proteins=2, n_residues=20, n_edges=60):
    data_list = []
    for _ in range(n_proteins):
        N, E = n_residues, n_edges
        data = Data(
            seq_idx=torch.randint(0, 21, (N,)),
            x_scalar=torch.randn(N, 27),      # 21 AA one-hot + 6 torsion (no RWSE)
            x_vec=torch.randn(N, 3, 3),
            edge_index=torch.randint(0, N, (2, E)),
            edge_scalar=torch.randn(E, 16),
            edge_vec=torch.randn(E, 1, 3),
        )
        data_list.append(data)
    return Batch.from_data_list(data_list)


CONFIG = {
    "num_layers": 2,
    "node_scalar_in": 91,      # 27 (aa+torsion) + 64 (walk_embed_dim)
    "node_vector_in": 3,
    "edge_scalar_in": 16,
    "edge_vector_in": 1,
    "hidden_scalar": 64,
    "hidden_vector": 8,
    "output_dim": 128,
    "dropout": 0.0,
    # Walk config (small values for test speed)
    "num_walks": 4,
    "walk_length": 5,
    "walk_embed_dim": 64,
    "walk_layers": 2,
    "walk_heads": 4,
}


def test_encoder_output_shape():
    model = ProStrEncoder(CONFIG)
    batch = make_fake_batch()
    total_N = batch.x_scalar.shape[0]
    emb = model(batch)
    assert emb.shape == (total_N, 128)


def test_encoder_output_is_finite():
    model = ProStrEncoder(CONFIG)
    batch = make_fake_batch()
    emb = model(batch)
    assert torch.all(torch.isfinite(emb))


def test_encoder_equivariant_embedding():
    """Embedding must not change when 3D vectors are rotated (rotation-invariant)."""
    model = ProStrEncoder(CONFIG)
    model.eval()
    batch = make_fake_batch(n_proteins=1)

    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    batch_rot = batch.clone()
    batch_rot.x_vec    = (Q @ batch.x_vec.transpose(-1, -2)).transpose(-1, -2)
    batch_rot.edge_vec = (Q @ batch.edge_vec.transpose(-1, -2)).transpose(-1, -2)

    with torch.no_grad():
        emb     = model(batch)
        emb_rot = model(batch_rot)

    # Walk embedding uses seq_idx (AA types) not 3D vectors → invariant
    # GVP output is invariant → total output invariant
    torch.testing.assert_close(emb, emb_rot, atol=1e-4, rtol=1e-3)


def test_encoder_parameter_count():
    model = ProStrEncoder(CONFIG)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 10_000, "Model seems too small"
    assert n_params < 100_000_000, "Model seems too large for config"


def test_encoder_return_hidden():
    """return_hidden=True must return (output, hidden) where hidden has shape (N, s_h+v_h)."""
    model = ProStrEncoder(CONFIG)
    batch = make_fake_batch(n_proteins=1)
    total_N = batch.x_scalar.shape[0]
    with torch.no_grad():
        out, hidden = model(batch, return_hidden=True)
    assert out.shape == (total_N, 128)
    s_h = CONFIG["hidden_scalar"]
    v_h = CONFIG["hidden_vector"]
    assert hidden.shape == (total_N, s_h + v_h)
```

- [ ] **Step 2: Run the updated test to verify it fails**

```bash
pytest tests/models/test_encoder.py -v
```

Expected: Tests fail because `ProStrEncoder` still has the old interface (no walk config keys).

- [ ] **Step 3: Rewrite `prostrencoder/models/encoder.py`**

```python
import torch
import torch.nn as nn
from prostrencoder.models.gvp import GVP, GVPConv
from prostrencoder.models.walk_sampler import WalkSampler
from prostrencoder.models.walk_encoder import WalkEncoder


class ProStrEncoder(nn.Module):
    """
    GVP-GNN protein structure encoder with CyberGFM walk-sequence embedding.

    Input (PyG Batch / Data):
        seq_idx    (N,) int64          — AA indices 0–20
        x_scalar   (N, 27) float32     — AA one-hot (21) + backbone torsion (6)
        x_vec      (N, 3, 3) float32   — local backbone frame (3 orthonormal vectors)
        edge_index (2, E) int64
        edge_scalar (E, 16) float32    — Gaussian RBF of Cα–Cα distances
        edge_vec   (E, 1, 3) float32   — unit direction vectors src→dst

    Forward pass:
        1. WalkSampler samples K walks per residue → token seqs (N, K, L+2)
        2. WalkEncoder encodes token seqs → walk_emb (N, walk_embed_dim)
        3. node_s = cat([x_scalar, walk_emb], dim=-1)  → (N, 91)
        4. GVP-GNN processes node_s and node_v through L layers
        5. Output: scalars + vector norms → projection → (N, output_dim)

    Output: (N, output_dim) rotation-invariant per-residue embeddings.
    The walk embedding is invariant (derived from AA types, not 3D coords).
    The GVP output is invariant by construction. Combined output is invariant.
    """

    def __init__(self, config: dict):
        super().__init__()
        s_n  = config["node_scalar_in"]    # 91 = 27 + walk_embed_dim
        v_n  = config["node_vector_in"]    # 3
        s_e  = config["edge_scalar_in"]    # 16
        v_e  = config["edge_vector_in"]    # 1
        s_h  = config["hidden_scalar"]     # 128
        v_h  = config["hidden_vector"]     # 16
        out  = config["output_dim"]        # 512
        L    = config["num_layers"]        # 3
        drop = config["dropout"]           # 0.1

        # ── CyberGFM-inspired walk components ────────────────────────────────
        self.walk_sampler = WalkSampler(
            num_walks=config["num_walks"],      # 8
            walk_length=config["walk_length"],  # 20
        )
        walk_seq_len = config["walk_length"] + 2   # CLS + walk_length+1 nodes
        self.walk_encoder = WalkEncoder(
            vocab_size=22,
            embed_dim=config["walk_embed_dim"],         # 64
            num_heads=config["walk_heads"],             # 4
            num_layers=config["walk_layers"],           # 4
            ff_dim=config["walk_embed_dim"] * 4,        # 256
            max_seq_len=walk_seq_len,
            dropout=drop,
        )

        # ── GVP-GNN backbone (unchanged) ─────────────────────────────────────
        self.node_in = GVP((s_n, v_n), (s_h, v_h))
        self.edge_in = GVP((s_e, v_e), (s_h // 4, v_e))

        s_e_h = s_h // 4
        self.layers = nn.ModuleList([
            GVPConv(
                node_in_dims=(s_h, v_h),
                edge_in_dims=(s_e_h, v_e),
                node_out_dims=(s_h, v_h),
                drop_rate=drop,
            )
            for _ in range(L)
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(s_h + v_h),
            nn.Linear(s_h + v_h, out),
            nn.ReLU(),
            nn.Linear(out, out),
        )

    def forward(self, batch, return_hidden: bool = False):
        """
        batch        : PyG Batch or Data.
        return_hidden: if True, returns (output, hidden) where
                       hidden is (N, hidden_scalar + hidden_vector).
        """
        node_v     = batch.x_vec          # (N, 3, 3)
        edge_index = batch.edge_index     # (2, E)
        edge_s     = batch.edge_scalar    # (E, 16)
        edge_v     = batch.edge_vec       # (E, 1, 3)

        # ── Walk-sequence embedding (CyberGFM-inspired) ───────────────────────
        walk_tokens = self.walk_sampler(edge_index, batch.seq_idx)  # (N, K, L+2)
        walk_emb    = self.walk_encoder(walk_tokens)                 # (N, 64)

        # Concatenate with scalar node features
        node_s = torch.cat([batch.x_scalar, walk_emb], dim=-1)      # (N, 91)

        # ── GVP-GNN message passing ───────────────────────────────────────────
        node_s, node_v = self.node_in(node_s, node_v)
        edge_s, edge_v = self.edge_in(edge_s, edge_v)

        for layer in self.layers:
            node_s, node_v = layer(node_s, node_v, edge_index, edge_s, edge_v)

        # Rotation-invariant output: scalar features + vector norms
        v_norms = torch.norm(node_v, dim=-1)           # (N, v_h)
        hidden  = torch.cat([node_s, v_norms], dim=-1) # (N, s_h + v_h)

        output = self.out_proj(hidden)
        if return_hidden:
            return output, hidden
        return output
```

- [ ] **Step 4: Run the encoder tests to verify they pass**

```bash
pytest tests/models/test_encoder.py -v
```

Expected: All 5 tests PASS (4 original + 1 new `test_encoder_return_hidden`).

- [ ] **Step 5: Verify all other model tests are still green**

```bash
pytest tests/models/ -v
```

Expected: All tests in `test_walk_sampler.py`, `test_walk_encoder.py`, `test_gvp.py`, `test_rwse.py`, and `test_encoder.py` PASS (existing GVP/RWSE tests untouched).

- [ ] **Step 6: Commit**

```bash
git add prostrencoder/models/encoder.py tests/models/test_encoder.py
git commit -m "feat: integrate WalkSampler + WalkEncoder into ProStrEncoder (Option C)"
```

---

### Task 4: Update Preprocessing, Dataset Test, and Config

**Files:**
- Modify: `scripts/preprocess.py`
- Modify: `tests/data/test_dataset.py`
- Modify: `configs/default.yaml`

- [ ] **Step 1: Update `configs/default.yaml`**

Replace the entire file with:

```yaml
data:
  pdb_dir: data/pdb
  processed_dir: data/processed
  k_neighbors: 30

model:
  num_layers: 3
  node_scalar_in: 91      # 27 (21 AA + 6 torsion) + 64 (walk embedding)
  node_vector_in: 3       # backbone frame vectors
  edge_scalar_in: 16      # RBF distances
  edge_vector_in: 1       # direction vector
  hidden_scalar: 128
  hidden_vector: 16
  output_dim: 512
  dropout: 0.1
  # Walk-sequence encoder (CyberGFM-inspired, Option C)
  num_walks: 8
  walk_length: 20
  walk_embed_dim: 64
  walk_layers: 4
  walk_heads: 4

training:
  batch_size: 32
  max_epochs: 100
  lr: 0.0001
  weight_decay: 0.01
  mask_rate: 0.15
  grad_clip: 1.0
  checkpoint_dir: checkpoints
  log_every: 50
```

- [ ] **Step 2: Update `scripts/preprocess.py`**

Replace the entire file with:

```python
"""
Convert a directory of PDB files to preprocessed PyG Data objects.

x_scalar stores only AA one-hot (21) + torsion angles (6) = 27 dims.
The 64-dim walk embedding is computed online inside ProStrEncoder.forward()
and is no longer stored in the .pt file.

Usage:
    python scripts/preprocess.py --pdb_dir data/pdb --out_dir data/processed --k 30
"""
import argparse
import os
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from prostrencoder.data.parser import parse_structure
from prostrencoder.data.graph_builder import build_knn_graph
from prostrencoder.data.features import (
    aa_one_hot, compute_backbone_frame, compute_torsion_angles,
    rbf_encoding, compute_edge_directions,
)


def preprocess_one(pdb_path: str, k: int) -> Data:
    parsed = parse_structure(pdb_path)
    if len(parsed["seq_idx"]) < 4:
        raise ValueError(f"Too few residues in {pdb_path}")

    edge_index, edge_dist = build_knn_graph(parsed["ca_coords"], k=k)

    onehot  = aa_one_hot(parsed["seq_idx"])                         # (N, 21)
    torsion = compute_torsion_angles(parsed["backbone_coords"])      # (N, 6)
    frame   = compute_backbone_frame(parsed["backbone_coords"])      # (N, 3, 3)
    rbf     = rbf_encoding(edge_dist)                                # (E, 16)
    dirs    = compute_edge_directions(
                  parsed["ca_coords"], edge_index, edge_dist)        # (E, 3)

    x_scalar = np.concatenate([onehot, torsion], axis=1).astype(np.float32)  # (N, 27)
    e_vec    = dirs[:, np.newaxis, :].astype(np.float32)                      # (E, 1, 3)

    return Data(
        seq_idx=torch.from_numpy(parsed["seq_idx"]),
        x_scalar=torch.from_numpy(x_scalar),
        x_vec=torch.from_numpy(frame),
        edge_index=torch.from_numpy(edge_index),
        edge_scalar=torch.from_numpy(rbf),
        edge_vec=torch.from_numpy(e_vec),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--k",       type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pdb_files = [f for f in os.listdir(args.pdb_dir)
                 if f.endswith((".pdb", ".cif"))]

    skipped = 0
    for fname in tqdm(pdb_files, desc="Preprocessing"):
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(args.out_dir, stem + ".pt")
        if os.path.exists(out_path):
            continue
        try:
            data = preprocess_one(os.path.join(args.pdb_dir, fname), args.k)
            torch.save(data, out_path)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            skipped += 1

    print(f"Done. Skipped {skipped}/{len(pdb_files)} files.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Update `tests/data/test_dataset.py`**

The fixture builds a Data object that mirrors what `preprocess_one()` produces.
`x_scalar` is now 27-dim (was 43). Replace the entire file with:

```python
import os
import numpy as np
import torch
import pytest
from torch_geometric.data import Data
from prostrencoder.data.dataset import ProteinDataset

MINIMAL_PDB = """\
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.526  10.000  10.000  1.00  0.00           C
ATOM      3  C   ALA A   1      12.000  11.400  10.000  1.00  0.00           C
ATOM      4  O   ALA A   1      11.200  12.200  10.000  1.00  0.00           O
ATOM      5  N   GLY A   2      13.300  11.700  10.000  1.00  0.00           N
ATOM      6  CA  GLY A   2      13.800  13.100  10.000  1.00  0.00           C
ATOM      7  C   GLY A   2      15.300  13.100  10.000  1.00  0.00           C
ATOM      8  O   GLY A   2      15.900  12.000  10.000  1.00  0.00           O
ATOM      9  N   LEU A   3      15.900  14.300  10.000  1.00  0.00           N
ATOM     10  CA  LEU A   3      17.400  14.300  10.000  1.00  0.00           C
ATOM     11  C   LEU A   3      17.900  15.700  10.000  1.00  0.00           C
ATOM     12  O   LEU A   3      17.100  16.600  10.000  1.00  0.00           O
END
"""


@pytest.fixture
def processed_dir(tmp_path):
    """Build one preprocessed .pt file matching the new preprocess_one() output."""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text(MINIMAL_PDB)

    from prostrencoder.data.parser import parse_structure
    from prostrencoder.data.graph_builder import build_knn_graph
    from prostrencoder.data.features import (
        aa_one_hot, compute_backbone_frame, compute_torsion_angles,
        rbf_encoding, compute_edge_directions,
    )

    parsed = parse_structure(str(pdb_path))
    edge_index, edge_dist = build_knn_graph(parsed["ca_coords"], k=30)
    onehot  = aa_one_hot(parsed["seq_idx"])
    torsion = compute_torsion_angles(parsed["backbone_coords"])
    frame   = compute_backbone_frame(parsed["backbone_coords"])
    rbf     = rbf_encoding(edge_dist)
    dirs    = compute_edge_directions(parsed["ca_coords"], edge_index, edge_dist)

    x_scalar = np.concatenate([onehot, torsion], axis=1).astype(np.float32)  # (N, 27)
    e_vec    = dirs[:, np.newaxis, :].astype(np.float32)                      # (E, 1, 3)

    data = Data(
        seq_idx=torch.from_numpy(parsed["seq_idx"]),
        x_scalar=torch.from_numpy(x_scalar),
        x_vec=torch.from_numpy(frame),
        edge_index=torch.from_numpy(edge_index),
        edge_scalar=torch.from_numpy(rbf),
        edge_vec=torch.from_numpy(e_vec),
    )
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()
    torch.save(data, proc_dir / "test.pt")
    return str(proc_dir)


def test_dataset_length(processed_dir):
    ds = ProteinDataset(processed_dir)
    assert len(ds) == 1


def test_dataset_item_keys(processed_dir):
    ds = ProteinDataset(processed_dir)
    item = ds[0]
    assert hasattr(item, "seq_idx")
    assert hasattr(item, "x_scalar")
    assert hasattr(item, "x_vec")
    assert hasattr(item, "edge_index")
    assert hasattr(item, "edge_scalar")
    assert hasattr(item, "edge_vec")


def test_dataset_feature_dims(processed_dir):
    ds = ProteinDataset(processed_dir)
    item = ds[0]
    N = item.seq_idx.shape[0]
    assert item.x_scalar.shape == (N, 27)   # 21 AA + 6 torsion (no RWSE)
    assert item.x_vec.shape == (N, 3, 3)
    assert item.edge_scalar.shape[1] == 16
    assert item.edge_vec.shape[1:] == (1, 3)
```

- [ ] **Step 4: Run the dataset tests**

```bash
pytest tests/data/test_dataset.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Run the full test suite to confirm nothing regressed**

```bash
pytest tests/ -v
```

Expected: All tests PASS. Count should be ≥ 54 (48 original + 6 walk_sampler + 6 walk_encoder − 1 old encoder test + 5 new encoder tests = ~54 total, exact count depends on test additions).

Example passing output:

```
tests/data/test_dataset.py          ... 3 passed
tests/data/test_features.py         ... 9 passed
tests/data/test_graph_builder.py    ... 8 passed
tests/data/test_parser.py           ... 5 passed
tests/models/test_encoder.py        ... 5 passed
tests/models/test_gvp.py            ... 6 passed
tests/models/test_rwse.py           ... 6 passed
tests/models/test_walk_encoder.py   ... 6 passed
tests/models/test_walk_sampler.py   ... 6 passed
tests/training/test_objectives.py   ... 7 passed
```

- [ ] **Step 6: Smoke-test the full training pipeline with synthetic data**

```bash
python - <<'EOF'
import os, torch, yaml
from torch_geometric.data import Data
from prostrencoder.data.dataset import ProteinDataset
from prostrencoder.training.trainer import Trainer

# Create synthetic .pt files matching new 27-dim x_scalar
for split in ["train", "val"]:
    os.makedirs(f"data/processed_v2/{split}", exist_ok=True)
    for i in range(4):
        N, E = 30, 90
        d = Data(
            seq_idx=torch.randint(0, 20, (N,)),
            x_scalar=torch.randn(N, 27),          # 27-dim, no RWSE
            x_vec=torch.randn(N, 3, 3),
            edge_index=torch.randint(0, N, (2, E)),
            edge_scalar=torch.randn(E, 16),
            edge_vec=torch.randn(E, 1, 3),
        )
        torch.save(d, f"data/processed_v2/{split}/prot_{i}.pt")

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)
config["training"]["max_epochs"] = 2
config["training"]["batch_size"] = 2
config["training"]["log_every"] = 10

trainer = Trainer(
    config,
    ProteinDataset("data/processed_v2/train"),
    ProteinDataset("data/processed_v2/val"),
    device="cpu",
)
trainer.fit()
print("Smoke test passed.")
EOF
```

Expected output: 2 epochs complete, loss values printed, checkpoint saved, `Smoke test passed.`

- [ ] **Step 7: Commit**

```bash
git add configs/default.yaml scripts/preprocess.py tests/data/test_dataset.py
git commit -m "feat: drop RWSE from preprocessing; x_scalar is now (N,27); update config for walk encoder"
```

---

## Self-Review

### Spec coverage check

| Requirement | Task |
|---|---|
| WalkSampler: K=8 walks of L=20, AA-type tokens, CLS prepended | Task 1 |
| WalkEncoder: 4-layer transformer, 64-dim, mean-pool → (N,64) | Task 2 |
| WalkSampler + WalkEncoder as submodules of ProStrEncoder | Task 3 |
| forward() concatenates walk_emb with x_scalar → 91-dim before GVP | Task 3 |
| Preprocessing drops compute_rwse; x_scalar → (N,27) | Task 4 |
| Config: node_scalar_in 43→91; add walk params; remove rwse_steps | Task 4 |
| All 48 existing tests remain green | Task 4 step 5 |

### Placeholder scan — none found

All code blocks are complete and runnable.

### Type consistency check

- `WalkSampler.forward(edge_index, seq_idx)` → `(N, K, L+2)` int64 — used identically in Task 1 (tests) and Task 3 (encoder)
- `WalkEncoder.forward(walk_tokens)` where `walk_tokens: (N, K, L)` → `(N, embed_dim)` float32 — defined in Task 2; consumed in Task 3 as `walk_tokens: (N, K, L+2)` ✓ (L+2 satisfies "L" parameter)
- `WalkSampler.CLS_TOKEN = 21` — referenced in Task 1 tests and is the class attribute defined in Task 1 implementation ✓
- `node_scalar_in: 91` — matches `27 (x_scalar) + 64 (walk_embed_dim)` throughout Tasks 3 and 4 ✓
- `walk_seq_len = config["walk_length"] + 2` in encoder — matches WalkSampler output length `walk_length + 2` ✓
- `x_scalar` shape: `(N, 27)` in preprocessed files (Task 4), `(N, 27)` in fake batches (Task 3), concatenated to `(N, 91)` inside `forward()` (Task 3) ✓

---

**Plan complete and saved to `docs/superpowers/plans/2026-06-01-cybergfm-option-c.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
