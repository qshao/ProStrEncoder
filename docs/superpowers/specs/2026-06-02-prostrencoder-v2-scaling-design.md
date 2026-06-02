# ProStrEncoder v2 — Scaling Design Spec

**Date:** 2026-06-02  
**Status:** Approved  
**Authors:** Research group discussion, recorded by Claude Code  

---

## Context and Goals

ProStrEncoder v1 is a 2.2M-parameter protein structure encoder combining GVP-GNN (local equivariant geometry) with a CyberGFM-inspired WalkEncoder (topology + sequence context). It is pre-trained with masked residue prediction on synthetic toy data and outputs 512-dim rotation-invariant per-residue embeddings.

**v2 goals:**
- Scale to ~85M–2.5B parameters
- Train on PDB (~220K experimental structures) with future AFDB augmentation
- Achieve strong performance on **function prediction** (GO terms, EC numbers, binding sites) and **sequence-structure design** (inverse folding, mutation effect prediction)
- Support **2–4 × H200 single-node** training with clean single-GPU fallback

**Compute target:** 2–4 × NVIDIA H200 (141 GB each), single node, DDP or FSDP

---

## Section 1: Architecture

### Overview

The v2 architecture adds a **global transformer** after the existing GVP-GNN local encoder. The forward pass has three stages:

```
Input (seq_idx, x_scalar, x_vec, edge_index, edge_scalar, edge_vec)
        │
        ▼
┌──────────────────────────────────────┐
│  WalkSampler + WalkEncoder           │
│  Samples K random walks per residue  │
│  (N, K, L+2) int64 → (N, walk_dim)  │
└──────────────────────────────────────┘
        │  concat with x_scalar (N, 27)
        ▼
┌──────────────────────────────────────┐
│  GVP-GNN Local Encoder               │
│  SE(3)-equivariant message passing   │
│  L_gvp layers, hidden_scalar dims    │
│  (N, node_scalar_in) → (N, d_gvp)   │
└──────────────────────────────────────┘
        │  invariant scalars (N, d_gvp)
        ▼
┌──────────────────────────────────────┐
│  Global Transformer                  │  ← NEW in v2
│  Flash Attention 2 + RoPE            │
│  L_tx layers, d_model width          │
│  (N, d_gvp) → (N, d_model)          │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  Output Heads                        │
│  • Masked residue head  (21-class)   │  pre-training only
│  • Inverse folding head (21-class)   │  pre-training only; NEW in v2
│  • Embedding projection (D_out)      │  kept at inference
└──────────────────────────────────────┘
```

### Rationale for Hybrid Design

The GVP layers handle 3D geometry with SE(3)-equivariance — they are well-validated, fast (O(E) in edges), and correctly encode local backbone structure. The global transformer operates on the invariant scalar output of GVP, so equivariance is not required there. This separation gives:

- **Local geometry** (secondary structure, backbone frame, residue packing): captured by GVP
- **Long-range context** (binding sites, allosteric pathways, domain interfaces): captured by global attention in O(1) layers regardless of protein length

The global transformer is standard `nn.TransformerEncoderLayer`-compatible architecture using Flash Attention 2 kernels — all GPU efficiency techniques (Flash Attention, FSDP, gradient checkpointing, `torch.compile`) apply without custom kernels.

### Global Transformer Details

The transformer uses:

- **Flash Attention 2** via `flash_attn.modules.mha.MHA` (dao-ailab, `pip install flash-attn`)
- **RoPE positional encoding** built into the Flash Attention MHA kernel (`rotary_emb_dim` parameter) — captures residue chain ordering, handles variable-length proteins, no separate positional embedding table
- **Sliding window attention** for long proteins: `window_size=(256, 256)` in Flash Attention MHA enables O(N·w) cost instead of O(N²) for proteins >1024 residues
- **Pre-norm** architecture (`norm_first=True`) for training stability at depth
- **GELU** activation in feed-forward layers

```python
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import GatedMlp

class GlobalTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_mult=4, dropout=0.1,
                 window_size=(-1, -1)):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MHA(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            causal=False,
            window_size=window_size,
            rotary_emb_dim=d_model // num_heads,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = GatedMlp(d_model, hidden_features=d_model * ff_mult,
                           activation=nn.GELU())

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.ff(self.norm2(x))
        return x
```

### Model Size Tiers

All tiers are controlled purely by config — same codebase, different YAML files.

| Tier | GVP layers | GVP hidden_scalar | GVP hidden_vector | Transformer layers | d_model | Walk dim | Approx params |
|------|-----------|-------------------|-------------------|--------------------|---------|----------|---------------|
| Small (v1) | 3 | 128 | 16 | — | — | 64 | ~2M |
| Medium | 6 | 256 | 32 | 12 | 512 | 128 | ~85M |
| Large | 8 | 512 | 64 | 24 | 1024 | 256 | ~860M |
| XL | 12 | 512 | 64 | 32 | 1536 | 256 | ~2.5B |

**GVP → transformer bridge:** a single `nn.Linear(d_gvp + v_h, d_model)` projects GVP's invariant output (scalar features concatenated with vector norms) to the transformer's d_model before the first transformer layer. This projection is shared across all tiers.

### Long-Protein Handling

- Default `max_len = 1024` residues per protein during training
- Proteins longer than `max_len` are randomly cropped to a contiguous window at each epoch (data augmentation)
- For the transformer, `window_size=(-1, -1)` (full attention) when N ≤ 1024; `window_size=(256, 256)` (sliding window) when N > 1024 — configurable via `attn_window_size` in YAML

### New Config Fields

```yaml
model:
  # GVP backbone (existing, scaled up per tier)
  num_layers: 6              # GVP conv layers
  hidden_scalar: 256
  hidden_vector: 32
  node_scalar_in: 155        # walk_dim(128) + 27 stored = 155 for Medium
  node_vector_in: 3
  edge_scalar_in: 16
  edge_vector_in: 1
  output_dim: 512

  # Walk encoder (scaled per tier)
  num_walks: 8
  walk_length: 20
  walk_embed_dim: 128        # Medium tier
  walk_layers: 6
  walk_heads: 8
  max_deg_cap: 30

  # Global transformer (NEW)
  tx_layers: 12              # number of transformer layers
  tx_d_model: 512            # transformer width; must match GVP bridge output
  tx_num_heads: 8
  tx_ff_mult: 4              # feed-forward multiplier (d_ff = tx_d_model * tx_ff_mult)
  tx_dropout: 0.1
  attn_window_size: -1       # -1 = full attention; positive int = sliding window per side
  gradient_checkpointing: true  # transformer layers only
```

---

## Section 2: Pre-training Objectives

Two objectives are used during joint training, applied to **alternating batches**:

### Objective 1 — Masked Residue Prediction (existing)

- 15% of residues masked: `x_scalar` zeroed, `seq_idx → MASK_AA` (token 22)
- Model predicts original amino acid type from structural + walk context
- Loss: cross-entropy over masked positions only
- Head: `nn.Linear(tx_d_model, 21)` applied to transformer output at masked positions

### Objective 2 — Inverse Folding (NEW)

- Full unmasked backbone structure as input (no masking)
- Model predicts the complete amino acid sequence at every position
- Loss: cross-entropy over all N positions
- Head: `nn.Linear(tx_d_model, 21)` — separate weights from masked residue head
- At inference: head is discarded; only embedding projection is kept

### Alternating Batch Strategy

Each batch is randomly assigned to one of two modes:

| Mode | Probability | Input | Loss |
|------|-------------|-------|------|
| A — Masked residue | 0.70 | Corrupted (15% masked) | CE on masked positions |
| B — Inverse folding | 0.30 | Full structure | CE on all positions |

This avoids running two forward passes per step (which would double memory and compute). The 70/30 split and the inverse folding loss weight λ are both configurable:

```yaml
training:
  mask_rate: 0.15
  inverse_fold_prob: 0.3     # fraction of batches in Mode B
  inverse_fold_weight: 1.0   # λ scaling factor on inverse folding loss
```

### What Each Objective Teaches

- **Mode A:** local structural context drives sequence identity — residue type is predictable from its geometric neighbourhood. Primary signal for structural understanding.
- **Mode B:** global backbone shape determines full sequence compatibility — the model must learn how residues co-adapt across the entire chain. Primary signal for design tasks.

---

## Section 3: Training Pipeline

### Two-Phase Schedule

#### Phase 1 — GVP Warm-up (5% of total steps)

- The transformer is instantiated at program start but **excluded from the optimizer** during Phase 1; its parameters receive no gradient updates
- The forward pass routes through GVP only; a temporary masked-residue head (reused from Phase 2) is applied directly to GVP output
- Objective: masked residue prediction only (Mode A)
- Single optimizer parameter group (GVP + WalkEncoder + masked residue head), cosine LR schedule starts from peak LR
- Purpose: GVP learns meaningful local geometry before transformer sees its output
- At warm-up end: save Phase 1 checkpoint; add transformer and inverse folding head parameters to the optimizer as a second group; continue without resetting the LR schedule

#### Phase 2 — Joint Training (remaining 95% of steps)

- All components train together: WalkEncoder + GVP + Global Transformer + both heads
- Alternating batch mode (70% Mode A / 30% Mode B)
- **Two optimizer parameter groups:**
  - Group 1 — GVP + WalkEncoder: `lr × gvp_lr_multiplier` (default 0.1)
  - Group 2 — Global transformer + both heads: `lr × 1.0`
- Same cosine LR schedule continues from Phase 1 (no reset)
- Gradient checkpointing active on transformer layers

### Optimizer and Schedule

```yaml
training:
  optimizer: adamw
  lr: 3e-4                   # peak LR for transformer (Group 2)
  weight_decay: 0.01
  grad_clip: 1.0
  lr_warmup_steps: 2000      # linear LR warm-up at start of training
  gvp_lr_multiplier: 0.1     # GVP group LR = lr × this value
  warmup_phase_fraction: 0.05  # Phase 1 duration as fraction of total steps
  max_epochs: 100
```

LR schedule: linear warm-up for `lr_warmup_steps` steps, then cosine decay to 0 over the remaining steps. Runs continuously across both phases.

### Device Strategy

Device configuration is set in YAML and validated at launch:

```yaml
training:
  # Options: "single_gpu" | "ddp" | "fsdp"
  device_strategy: single_gpu
  num_gpus: 1                # ignored when device_strategy=single_gpu
```

**Launch commands:**

```bash
# Single GPU (any tier up to Large with gradient checkpointing)
python scripts/train.py --config configs/medium.yaml

# Multi-GPU DDP — Medium or Large
torchrun --nproc_per_node=4 scripts/train.py --config configs/large.yaml

# Multi-GPU FSDP — XL (2.5B params)
torchrun --nproc_per_node=4 scripts/train.py --config configs/xl.yaml
```

The training script detects `LOCAL_RANK` from the environment:
- `device_strategy=single_gpu` + `LOCAL_RANK` set → raise clear error (launched with `torchrun` by mistake)
- `device_strategy=ddp` or `fsdp` + no `LOCAL_RANK` → raise clear error (must use `torchrun`)

**DDP** (Medium/Large, ≤860M params): `torch.nn.parallel.DistributedDataParallel` — lower overhead, simpler debugging.

**FSDP** (XL, 2.5B+ params): `torch.distributed.fsdp.FullyShardedDataParallel` — shards parameters, gradients, and optimizer state across GPUs. Checkpoints saved with `StateDictType.FULL_STATE_DICT` for portability.

### Memory and Precision

```yaml
training:
  precision: bf16             # "fp32" | "bf16" (required for flash_attn on H200)
  compile: true               # torch.compile on GVP forward pass
```

- **BF16 throughout** — H200 supports BF16 natively; better dynamic range than FP16; no GradScaler needed
- **Flash Attention 2** on transformer layers — 2–4× faster than native SDPA; built-in sliding window; BF16 required
- **Gradient checkpointing** on transformer layers only — GVP is shallow and cheap
- **`torch.compile`** on the GVP encoder forward pass — meaningful speedup on repeated graph operations; not applied to the transformer (Flash Attention already handles this)

### Checkpointing

- Save every `save_every_steps` steps + always save the best validation checkpoint
- Checkpoint contains: encoder state dict, both head state dicts, optimizer state, scheduler state, current step, config snapshot
- FSDP: use `FullStateDictConfig(offload_to_cpu=True)` to save consolidated checkpoint from rank 0

```yaml
training:
  checkpoint_dir: checkpoints
  save_every_steps: 1000
  log_every: 50
  wandb: false
  wandb_project: prostrencoder
```

---

## Section 4: Data Pipeline

### Layer 1 — PDB Quality Filtering

New script `scripts/download_pdb.py` fetches structures from RCSB and applies per-chain filters:

```yaml
data:
  pdb_dir: data/pdb
  processed_dir: data/processed
  k_neighbors: 30
  max_len: 1024              # residues; longer chains are randomly cropped at train time
  min_len: 40
  resolution_cutoff: 3.5    # Å — discard low-resolution X-ray structures
  min_chain_coverage: 0.9   # fraction of residues with all 4 backbone atoms present
  cluster_seqid: 0.30       # MMseqs2 sequence identity threshold for train/val split
```

**Filters applied per chain:**
1. Resolution ≤ 3.5 Å (X-ray) or cryo-EM (any resolution accepted)
2. ≥ 90% of residues have all four backbone atoms (N, CA, C, O)
3. Chain length within [min_len, max_len]
4. Redundancy reduction: 30% sequence identity clustering with MMseqs2 — one representative per cluster for training; prevents homolog leakage between train/val splits

### Layer 2 — AFDB Augmentation (future)

When AFDB structures are added:

```yaml
data:
  afdb_dir: data/afdb
  afdb_plddt_cutoff: 70     # only include residues with pLDDT ≥ 70
```

- AFDB structures preprocessed identically to PDB
- Per-residue pLDDT stored as additional field `plddt (N,)` in the Data object
- pLDDT used to weight the inverse folding loss: residues with pLDDT < 70 contribute zero loss weight (low-confidence predicted structure should not be treated as ground truth)

### Layer 3 — Preprocessing

`scripts/preprocess.py` parallelised with `multiprocessing.Pool` — each worker independently converts one PDB file to a `.pt` file using the existing `build_pyg_data` pipeline:

```yaml
data:
  preprocess_workers: 16    # parallel PDB → .pt workers
```

No changes to `build_pyg_data`, `parse_structure`, or `build_knn_graph`.

### Layer 4 — Train-time Loading

**For PDB scale (~220K structures):** existing `ProteinDataset` (random-access `.pt` files) with `num_workers=4`.

**For AFDB scale (millions of structures):** new `ShardedProteinDataset` using WebDataset format — `.pt` files packed into sequential `.tar` shards for efficient streaming I/O. Config selects which to use:

```yaml
data:
  dataset_format: random_access   # "random_access" | "sharded"
  shard_dir: data/shards          # used when dataset_format=sharded
```

**Random crop for long proteins:** proteins longer than `max_len` are randomly cropped to a contiguous window of `max_len` residues at each epoch — keeps batch shapes bounded and provides data augmentation.

**Token-budget batching for AFDB scale:** rather than fixed `batch_size`, group proteins by token budget to maximise GPU utilisation across variable-length chains:

```yaml
training:
  batch_size: 32             # used when dataset_format=random_access
  token_budget: 16384        # residues per batch; used when dataset_format=sharded
```

---

## Section 5: Infrastructure and Evaluation

### Flash Attention 2

Install: `pip install flash-attn --no-build-isolation` (requires CUDA 11.8+, PyTorch 2.0+).

The global transformer replaces `nn.MultiheadAttention` with `flash_attn.modules.mha.MHA`:

```python
from flash_attn.modules.mha import MHA

self.attn = MHA(
    embed_dim=d_model,
    num_heads=num_heads,
    dropout=dropout,
    causal=False,
    window_size=(window_size, window_size),  # -1 = full attention
    rotary_emb_dim=d_model // num_heads,     # RoPE built-in
)
```

Key benefits over PyTorch native `scaled_dot_product_attention` on H200:
- **2–4× faster** for typical protein sequence lengths (64–1024 residues)
- **Sliding window attention built-in** — replaces Longformer-style approximation; `window_size=(256, 256)` for proteins >1024 residues
- **RoPE built into the kernel** — no separate positional embedding table; handles variable lengths correctly
- **BF16 native** — no loss scaling, simpler training loop

### Evaluation Suite

Three evaluation levels:

**1. During training (every checkpoint):**
- Masked residue prediction **perplexity** on held-out PDB validation set
- Inverse folding **sequence recovery rate** on held-out PDB validation set
- Both computed on a single GPU without labels; fast enough to run each checkpoint

**2. Linear probe (post-training milestones):**
- Existing `eval_linear_probe.py` — frozen encoder → linear head on SCOP fold classification
- Primary quality metric for function prediction capability
- Run after each major training milestone (end of Phase 1, end of training, after fine-tuning)

**3. Inverse folding recovery (post-training):**
- New `eval_inverse_folding.py`
- Given held-out PDB structures, predict sequences with the inverse folding head
- Report native sequence recovery rate (fraction of positions correctly predicted)
- Target baseline: >35% recovery rate (ProteinMPNN achieves ~52%; a structure encoder pre-trained jointly is expected to start lower)

### Config File Structure

One YAML per model tier, each overriding only what changes from `default.yaml`:

```
configs/
  default.yaml     # Small / v1 — 2M params, single GPU (existing)
  medium.yaml      # 85M params, single GPU or 2×H200 DDP
  large.yaml       # 860M params, 2-4×H200 DDP
  xl.yaml          # 2.5B params, 4×H200 FSDP
```

### Monitoring and Logging

Training logs emit structured JSON lines per step (readable by TensorBoard or W&B):

```json
{
  "step": 1000, "epoch": 2, "mode": "A",
  "loss": 2.14, "perplexity": 8.5,
  "lr_tx": 2.8e-4, "lr_gvp": 2.8e-5,
  "gpu_mem_gb": 62.3, "residues_per_sec": 48000
}
```

W&B enabled with `--wandb` flag or `training.wandb: true` in config.

### Environment

Add to `environment.yml` or `requirements.txt`:

```
flash-attn>=2.5.0     # requires CUDA 11.8+, PyTorch 2.0+, H100/H200 or A100
mmseqs2               # for train/val sequence identity splitting
webdataset            # for sharded AFDB loading
wandb                 # optional, for experiment tracking
```

---

## Summary of All Design Decisions

| Dimension | Decision |
|-----------|----------|
| Architecture | GVP local encoder + global transformer (Option C) |
| Attention | Flash Attention 2 (dao-ailab) with RoPE + sliding window |
| Pre-training objectives | Masked residue prediction + inverse folding, alternating batches 70/30 |
| Training strategy | GVP warm-up (5%) → joint training (95%) with layer-wise LR |
| Multi-GPU | Config-driven: `single_gpu` / `ddp` / `fsdp`; validated at launch |
| Precision | BF16 throughout (required by flash_attn; native on H200) |
| Data — PDB | Quality-filtered (resolution, coverage, MMseqs2 clustering) |
| Data — AFDB | pLDDT-weighted loss; WebDataset sharding for streaming I/O |
| Long proteins | Random crop to max_len=1024 at train time; sliding window attention |
| Evaluation | Perplexity + sequence recovery (during training); SCOP linear probe + inverse folding recovery rate (post-training) |

---

## What Is Not Changed

The following v1 components are preserved without modification:

- `WalkSampler` — random walk sampling on protein graphs
- `build_pyg_data` — feature assembly (one-hot, torsion, RBF, backbone frame)
- `parse_structure` — PDB/mmCIF parsing
- `build_knn_graph` — k-NN graph construction
- `ProteinDataset` — random-access `.pt` loading (for PDB scale)
- `MaskedResidueLoss` — cross-entropy over masked positions
- `apply_masking` — BERT-style masking with MASK_AA token
- All existing tests — must continue to pass after v2 changes

---

## Open Questions (Deferred to Implementation)

1. **Warm-up duration sensitivity:** 5% of total steps is the recommended starting point; an ablation over {2%, 5%, 10%} at Medium scale before Large-scale training is advisable.
2. **Inverse folding batch fraction:** 30% Mode B is a starting point; if design task performance is weak, increase to 50%.
3. **GVP lr multiplier:** 0.1× is conservative; if GVP features stagnate during joint training, increase to 0.3×.
4. **AFDB sampling ratio:** when AFDB is added, start with a 1:1 PDB:AFDB ratio by sampling; adjust based on validation perplexity.
5. **Flash Attention availability fallback:** if `flash_attn` is not installed (e.g., CPU-only machine for testing), fall back to `torch.nn.functional.scaled_dot_product_attention` automatically with a warning.
