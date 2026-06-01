# ProStrEncoder

A foundational protein 3D structure encoder that produces per-residue embeddings from atomic coordinates using a Geometric Vector Perceptron Graph Neural Network (GVP-GNN) augmented with a CyberGFM-inspired walk-sequence transformer.

---

## 1. Introduction

Proteins carry out virtually every biological function in living cells, and their three-dimensional structure determines their activity. While sequence-based language models (e.g., ESM) have proven powerful, they treat proteins as strings of amino acids and cannot directly reason about 3D geometry. ProStrEncoder fills this gap: it reads atomic coordinates from a PDB or mmCIF file and outputs a **512-dimensional embedding vector for every residue**, encoding the local and global structural context in a rotation-invariant way.

### How it works

```
PDB / mmCIF file
       │
       ▼  parse_structure()
Residue arrays (Cα coords, backbone atoms, AA types)
       │
       ▼  build_knn_graph()  +  feature computation
PyG Data object (node features 27-dim scalar + 3 backbone-frame vectors,
                 edge features 16-dim RBF scalar + 1 direction vector)
       │
   ┌───┴─────────────────────────────────────────────────┐
   │ WalkSampler: K=8 random walks of L=20 steps each    │
   │   → AA-type token sequences  (N, 8, 22)             │
   │ WalkEncoder: 4-layer transformer, CLS mean-pool      │
   │   → 64-dim walk embedding    (N, 64)                 │
   │ cat([x_scalar, walk_emb])    (N, 91)                 │
   └───┬─────────────────────────────────────────────────┘
       │
       ▼  ProStrEncoder (3× GVP-GNN layers)
Per-residue embeddings  (N × 512)
```

**Key design choices:**

| Property | Detail |
|---|---|
| Graph topology | k-NN with k = 30, Cα–Cα Euclidean distance |
| Node scalars (stored) | 21-dim AA one-hot + 6-dim torsion angles (sin/cos φ,ψ,ω) = 27-dim |
| Walk embedding (online) | K=8 random walks of L=20 → WalkEncoder (4-layer transformer) → 64-dim |
| Total node scalars | 27 + 64 = 91-dim (concatenated inside `forward()`) |
| Node vectors | 3 × 3D backbone frame (N→CA, Gram-Schmidt CA→C, cross product) |
| Edge scalars | 16-dim Gaussian RBF of Cα–Cα distance (0–20 Å) |
| Edge vectors | 1 × 3D unit direction vector src→dst |
| GNN backbone | Geometric Vector Perceptrons (Jing et al. 2021) — SE(3)-equivariant |
| Walk embedding | Rotation-invariant: derived from AA-type token IDs, not 3D coords |
| Output | 512-dim rotation-invariant per-residue embeddings |
| Pre-training | Masked residue type prediction (BERT-style, 15% masking rate) |

The SE(3)-equivariance guarantee means the embeddings do not change when the protein is rotated or reflected in 3D space. This holds for both the GVP layers (equivariant by design) and the walk embedding (invariant because it depends only on AA-type token sequences, never on 3D coordinates).

---

## 2. Installation

### 2.1 Create a virtual environment

**Option A — conda (recommended for GPU users):**

```bash
conda create -n prostrencoder python=3.11
conda activate prostrencoder
```

**Option B — venv:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

### 2.2 Install PyTorch

Choose the command that matches your CUDA version from [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2.3 Install PyTorch Geometric

```bash
pip install torch-geometric
```

### 2.4 Install ProStrEncoder

Clone the repository and install the package in editable mode so that source edits are reflected immediately:

```bash
git clone <repository-url>
cd ProStrEncoder
pip install -e .
```

This installs all remaining dependencies (BioPython, SciPy, NumPy, PyYAML, tqdm) automatically.

### 2.5 Verify installation

```bash
python -c "
import prostrencoder
from prostrencoder.models.encoder import ProStrEncoder
print('ProStrEncoder installed successfully')
"
```

### 2.6 Run the test suite

```bash
pytest tests/ -v
```

Expected: **61 tests pass** in under 5 seconds.

---

## 3. Supported Modules

### 3.1 Data pipeline — `prostrencoder.data`

| Module | Public API | Description |
|---|---|---|
| `prostrencoder.data.parser` | `parse_structure(filepath)` | Parse a `.pdb` or `.cif` file into per-residue numpy arrays: `seq_idx` (N,), `ca_coords` (N,3), `backbone_coords` (N,4,3) |
| `prostrencoder.data.graph_builder` | `build_knn_graph(ca_coords, k=30)` | Build a directed k-NN graph from Cα coordinates; returns `edge_index` (2,E) int64 and `edge_dist` (E,) float32 |
| `prostrencoder.data.features` | `aa_one_hot(seq_idx)` | One-hot encode 21-class AA vocabulary → (N,21) float32 |
| | `compute_backbone_frame(backbone_coords)` | Local orthonormal frame per residue → (N,3,3) float32 |
| | `compute_torsion_angles(backbone_coords)` | sin/cos of φ,ψ,ω backbone torsion angles → (N,6) float32 |
| | `rbf_encoding(distances, num_rbf=16)` | Gaussian RBF encoding of distances → (E,16) float32 |
| | `compute_edge_directions(ca_coords, edge_index, edge_dist)` | Unit direction vectors src→dst → (E,3) float32 |
| `prostrencoder.data.dataset` | `ProteinDataset(processed_dir)` | PyTorch `Dataset` that loads preprocessed `.pt` files produced by `scripts/preprocess.py` |

### 3.2 Model components — `prostrencoder.models`

| Module | Public API | Description |
|---|---|---|
| `prostrencoder.models.walk_sampler` | `WalkSampler(num_walks=8, walk_length=20)` | Samples K random walks per residue on the protein k-NN graph; converts node indices to AA-type token sequences with a CLS token prepended. Output: `(N, K, L+2)` int64. No trainable parameters. |
| `prostrencoder.models.walk_encoder` | `WalkEncoder(vocab_size=22, embed_dim=64, ...)` | 4-layer pre-norm transformer that encodes walk-token sequences → 64-dim per-residue embedding via CLS mean-pool over K walks. Rotation-invariant (token IDs only). |
| `prostrencoder.models.gvp` | `GVP(in_dims, out_dims)` | Geometric Vector Perceptron layer. `in_dims = (s_in, v_in)`, `out_dims = (s_out, v_out)`. SE(3)-equivariant. |
| | `GVPConv(node_in_dims, edge_in_dims, node_out_dims)` | One round of GVP-based message passing with sum aggregation and residual connection |
| `prostrencoder.models.encoder` | `ProStrEncoder(config)` | Full encoder: WalkSampler + WalkEncoder online → 91-dim scalars → GVP input projections → L × GVPConv → invariant output projection → (N, output_dim) |
| `prostrencoder.models.gnn` | `ResidueHead(in_dim, num_classes=21)` | Single linear head for masked residue type prediction (21-class) |
| `prostrencoder.models.rwse` | `compute_rwse(edge_index, num_nodes, walk_length=16)` | Legacy: random walk landing probabilities (replaced by WalkEncoder; kept for reference) |

### 3.3 Training — `prostrencoder.training`

| Module | Public API | Description |
|---|---|---|
| `prostrencoder.training.objectives` | `apply_masking(seq_idx, x_scalar, mask_rate=0.15)` | BERT-style masking: zeros out scalar features of randomly selected residues; returns `(masked_scalar, mask)` |
| | `MaskedResidueLoss()` | Cross-entropy loss over masked positions only; returns 0.0 when no residues are masked |
| `prostrencoder.training.trainer` | `Trainer(config, train_dataset, val_dataset, device)` | Full pre-training loop: AdamW optimizer, cosine LR schedule, gradient clipping, checkpoint saving |

### 3.4 Scripts

| Script | Purpose |
|---|---|
| `scripts/preprocess.py` | Convert a directory of PDB/mmCIF files to preprocessed `.pt` graph files |
| `scripts/train.py` | Pre-train ProStrEncoder via masked residue prediction |
| `scripts/eval_linear_probe.py` | Evaluate frozen embeddings on a labelled classification task (linear probe) |

---

## 4. Usage

### 4.1 Preprocess protein structures

Before training, convert PDB files to graph objects. This is a one-time offline step.

```bash
python scripts/preprocess.py \
    --pdb_dir  data/pdb \
    --out_dir  data/processed \
    --k        30
```

Each `.pdb` or `.cif` file in `data/pdb/` produces a corresponding `.pt` file in `data/processed/`. The `.pt` file is a `torch_geometric.data.Data` object with the following fields:

| Field | Shape | dtype | Content |
|---|---|---|---|
| `seq_idx` | (N,) | int64 | AA index (0–19 standard, 20 = UNK) |
| `x_scalar` | (N, 27) | float32 | AA one-hot (21) + torsion angles (6) |
| `x_vec` | (N, 3, 3) | float32 | Backbone frame vectors |
| `edge_index` | (2, E) | int64 | Graph edges |
| `edge_scalar` | (E, 16) | float32 | RBF distance encoding |
| `edge_vec` | (E, 1, 3) | float32 | Direction vectors |

> **Note:** The 64-dim walk-sequence embedding is **not stored** on disk. It is computed online by `WalkSampler` + `WalkEncoder` inside `ProStrEncoder.forward()` during every training step and inference call, using the `edge_index` and `seq_idx` fields already present in each `.pt` file.

### 4.2 Pre-train the encoder

Edit `configs/default.yaml` to set paths and hyperparameters, then:

```bash
python scripts/train.py \
    --config    configs/default.yaml \
    --train_dir data/processed/train \
    --val_dir   data/processed/val \
    --device    cuda
```

Training prints per-epoch loss and saves the best checkpoint (lowest validation loss) to the `checkpoints/` directory.

**Key hyperparameters in `configs/default.yaml`:**

```yaml
model:
  num_layers:      3      # number of GVP-GNN message-passing rounds
  node_scalar_in:  91     # 27 (stored) + 64 (walk embedding, online)
  hidden_scalar:   128    # GVP scalar feature width
  hidden_vector:   16     # number of 3D vector features per node
  output_dim:      512    # final embedding dimension
  dropout:         0.1
  # Walk-sequence encoder (CyberGFM-inspired)
  num_walks:       8      # random walks per residue
  walk_length:     20     # steps per walk
  walk_embed_dim:  64     # transformer hidden dim and output dim
  walk_layers:     4      # transformer encoder layers
  walk_heads:      4      # attention heads

training:
  max_epochs:  100
  batch_size:  32
  lr:          0.0001   # AdamW learning rate
  mask_rate:   0.15     # fraction of residues masked per protein
```

### 4.3 Embed a protein in Python

```python
import torch
import yaml
import numpy as np
from torch_geometric.data import Data
from prostrencoder.data.parser import parse_structure
from prostrencoder.data.graph_builder import build_knn_graph
from prostrencoder.data.features import (
    aa_one_hot, compute_backbone_frame, compute_torsion_angles,
    rbf_encoding, compute_edge_directions,
)
from prostrencoder.models.encoder import ProStrEncoder

# ── 1. Load config and model ─────────────────────────────────────────────────
with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

encoder = ProStrEncoder(config["model"])
ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=True)
encoder.load_state_dict(ckpt["encoder"])
encoder.eval()

# ── 2. Parse PDB file ────────────────────────────────────────────────────────
parsed = parse_structure("my_protein.pdb")

# ── 3. Build graph and compute features ──────────────────────────────────────
edge_index, edge_dist = build_knn_graph(parsed["ca_coords"], k=30)

# x_scalar is 27-dim: AA one-hot (21) + backbone torsion angles (6)
# The 64-dim walk embedding is computed inside encoder.forward() — not stored here.
x_scalar = np.concatenate([
    aa_one_hot(parsed["seq_idx"]),                          # (N, 21)
    compute_torsion_angles(parsed["backbone_coords"]),      # (N, 6)
], axis=1).astype(np.float32)                              # (N, 27)

x_vec  = compute_backbone_frame(parsed["backbone_coords"])            # (N, 3, 3)
e_scal = rbf_encoding(edge_dist)                                      # (E, 16)
e_vec  = compute_edge_directions(parsed["ca_coords"],
                                 edge_index, edge_dist)[:, None, :]   # (E, 1, 3)

# ── 4. Build PyG Data object ─────────────────────────────────────────────────
data = Data(
    seq_idx=torch.from_numpy(parsed["seq_idx"]),   # needed by WalkSampler
    x_scalar=torch.from_numpy(x_scalar),
    x_vec=torch.from_numpy(x_vec),
    edge_index=torch.from_numpy(edge_index),
    edge_scalar=torch.from_numpy(e_scal),
    edge_vec=torch.from_numpy(e_vec),
)

# ── 5. Run encoder ───────────────────────────────────────────────────────────
# WalkSampler + WalkEncoder run automatically inside forward():
#   walk_tokens = WalkSampler(edge_index, seq_idx)   → (N, 8, 22)
#   walk_emb    = WalkEncoder(walk_tokens)            → (N, 64)
#   node_s      = cat([x_scalar, walk_emb])           → (N, 91) → GVP layers
with torch.no_grad():
    embeddings = encoder(data)           # (N, 512)

print(f"Residues: {len(parsed['seq_idx'])}")
print(f"Embedding shape: {embeddings.shape}")   # e.g. torch.Size([142, 512])
```

### 4.4 Batch inference

For multiple proteins, stack `Data` objects into a `Batch`:

```python
from torch_geometric.data import Batch

# ... build data1, data2, ... as above
batch = Batch.from_data_list([data1, data2, data3])

with torch.no_grad():
    embeddings = encoder(batch)   # (N_total_residues, 512)
```

The `batch.batch` tensor maps each residue back to its source protein, enabling protein-level pooling:

```python
from scripts.eval_linear_probe import mean_pool

protein_emb = mean_pool(embeddings, batch.batch)  # (3, 512) — one per protein
```

### 4.5 Evaluate on a downstream task (linear probe)

Label your preprocessed `.pt` files with a `label` field (integer class), then:

```bash
python scripts/eval_linear_probe.py \
    --checkpoint  checkpoints/best.pt \
    --train_dir   data/scop/train \
    --test_dir    data/scop/test \
    --config      configs/default.yaml \
    --num_classes 1195 \
    --device      cuda
```

The script freezes the encoder, mean-pools residue embeddings to protein level, trains a logistic regression head (AdamW, 100 epochs), and prints test accuracy every 20 epochs.

---

## Project Structure

```
ProStrEncoder/
├── prostrencoder/
│   ├── data/
│   │   ├── parser.py          # PDB/mmCIF parser
│   │   ├── graph_builder.py   # k-NN graph construction
│   │   ├── features.py        # node/edge feature functions
│   │   └── dataset.py         # PyG ProteinDataset
│   ├── models/
│   │   ├── walk_sampler.py    # WalkSampler: graph → AA-token walks (CyberGFM)
│   │   ├── walk_encoder.py    # WalkEncoder: transformer walk → 64-dim embedding
│   │   ├── gvp.py             # GVP layer + GVPConv (SE(3)-equivariant)
│   │   ├── encoder.py         # ProStrEncoder (walk + GVP-GNN full model)
│   │   ├── gnn.py             # ResidueHead (prediction head)
│   │   └── rwse.py            # Legacy RWSE (kept for reference)
│   └── training/
│       ├── objectives.py      # apply_masking, MaskedResidueLoss
│       └── trainer.py         # Trainer (pre-training loop)
├── scripts/
│   ├── preprocess.py          # PDB → .pt batch conversion (27-dim x_scalar)
│   ├── train.py               # training entry point
│   └── eval_linear_probe.py   # linear probe evaluation
├── configs/
│   └── default.yaml           # hyperparameters (incl. walk encoder config)
├── docs/
│   └── cybergfm-integration-options.md  # design document: Options A/B/C
├── tests/                     # 61 unit tests
├── setup.py
└── requirements.txt
```

---

## References

- Jing, B., Eismann, S., Suriana, P., Townshend, R. J. L., & Dror, R. (2021). **Learning from Protein Structure with Geometric Vector Perceptrons.** *ICLR 2022.* [arXiv:2009.01411](https://arxiv.org/abs/2009.01411)
- Rampášek, L., Galkin, M., Dwivedi, V. P., Lim, W. L., Wolf, G., & Beaini, D. (2022). **Recipe for a General, Powerful, Scalable Graph Transformer.** *NeurIPS 2022.* [arXiv:2205.12454](https://arxiv.org/abs/2205.12454) *(RWSE positional encoding)*
- CyberGFM (2024). **Finetuning Graph Foundation Models to Detect Lateral Movement in Enterprise Networks.** [github.com/cybermonic/CyberGFM](https://github.com/cybermonic/CyberGFM) *(walk-sequence tokenizer + BERT pre-training, adapted for protein residue graphs)*
