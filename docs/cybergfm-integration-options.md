# CyberGFM Integration Options for ProStrEncoder

This document describes three approaches for incorporating the GNN technology from
[CyberGFM](https://github.com/cybermonic/CyberGFM) into ProStrEncoder. Each option
is self-contained and can be evaluated independently.

---

## Background

### What CyberGFM contributes

CyberGFM is a graph foundation model originally designed for cybersecurity network
graphs. Its core innovations are:

| Component | Description |
|---|---|
| **Walk tokenizer** | Serializes a graph into a 1D token sequence via Eulerian path traversal or random walks, with special tokens (`SOS`, `EOS`, `JUMP`, `MASK`) |
| **GCN embedding** (`gnn_bert.py`) | Multi-layer `GCNConv` (PyG) produces initial geometry-aware node embeddings that replace a standard lookup table |
| **BERT transformer** (`bert.py`) | 12-layer, 768-dim transformer pretrained by predicting 15% masked tokens in the walk sequence |
| **Temporal random walker** (`sampler.py`) | Generates walks that respect edge timestamps; the static variant uses `torch_cluster.random_walk` |

The central idea: **serialize the graph → pretrain with masked token prediction → read off contextual node embeddings from transformer hidden states**.

### What ProStrEncoder currently uses

| Component | Role |
|---|---|
| `compute_rwse()` | 16-dim random-walk landing probabilities as node positional encoding |
| `GVP` / `GVPConv` | SE(3)-equivariant message passing operating on (scalar, vector) feature pairs |
| `ProStrEncoder` | 3 × GVPConv layers → 512-dim rotation-invariant per-residue embedding |
| Masked residue prediction | BERT-style pre-training objective on AA type |

### Key adaptation constraints

CyberGFM was built for **categorical, temporally-ordered network graphs**. Protein
residue graphs differ in three important ways:

1. **Continuous 3D geometry** — backbone torsion angles, RBF distances, and direction
   vectors are continuous, not categorical tokens.
2. **Rotational invariance requirement** — embeddings should not change when the protein
   is rotated or reflected. GCN applied directly to 3D coordinates breaks this.
3. **No temporal ordering** — edges have no timestamps; temporal random walks do not apply.

The walk tokenizer and BERT transfer to proteins cleanly. The GCN embedding requires
care: 3D coordinates must enter only via invariant features (distances, angles) to
preserve the rotational invariance guarantee.

---

## Option A — Stack: GVP-GNN → Walk-Sequence Transformer

### Concept

Keep the GVP-GNN backbone for equivariant local geometry processing. After the GVP
layers produce per-residue embeddings, use CyberGFM's walk tokenizer and a BERT
transformer to capture **long-range context** across the protein graph.

```
Input: Cα coordinates + backbone atoms + AA types
        │
        ▼  Feature extraction (unchanged)
    (N, 43) scalar features + (N, 3, 3) backbone frame vectors
        │
        ▼  GVP-GNN (3 × GVPConv, existing)  ← equivariant local geometry
    Per-residue embeddings  (N, 512)
        │
        ▼  Walk tokenizer (CyberGFM)
    K random walks of length L per residue  →  token sequences
        │
        ▼  BERT transformer (12 layers, 768-dim)  ← long-range context
    Contextualized per-residue embeddings  (N, 768)
        │
        ▼  Projection head
    Final per-residue embeddings  (N, 512)
```

### What changes in the codebase

| File | Change |
|---|---|
| `prostrencoder/models/walk_sampler.py` | New — adapted from CyberGFM `sampler.py`; static random walks on protein k-NN graph |
| `prostrencoder/models/walk_encoder.py` | New — BERT transformer (12 layers, 768-dim) operating on walk-token sequences |
| `prostrencoder/models/encoder.py` | Extended — GVP output fed as initial embeddings into `WalkEncoder`; final projection head added |
| `prostrencoder/training/trainer.py` | Extended — walk sampling during the training forward pass |
| `configs/default.yaml` | Add `walk_length`, `num_walks`, `walk_hidden`, `walk_layers` fields |

### Pros

- **Most expressive**: GVP handles equivariant local 3D geometry; transformer handles
  non-local topological context. The two are complementary.
- **Preserves all existing guarantees**: SE(3)-equivariance is maintained because GVP
  runs first and the transformer operates on its invariant scalar output.
- **Matches CyberGFM most closely**: the full walk-tokenizer + BERT stack is imported
  with minimal modification.
- **Best long-term quality**: combining local equivariant and global transformer
  representations is the direction of leading protein structure models.

### Cons

- **Large model**: full 12-layer BERT adds ~85 M parameters on top of the GVP-GNN.
- **Training complexity**: two architectural components with potentially different
  learning dynamics; may need separate learning rates or a warmup schedule.
- **Memory cost**: storing K walks per residue per batch can be significant for long
  proteins.
- **Highest implementation effort**: two new modules, changes to the training loop,
  walk batching logic.

### Recommended when

- Maximum embedding quality is the priority.
- GPU memory ≥ 24 GB is available.
- The project has entered a stable phase and can absorb a major architecture change.
- This is the natural **v2 target** after Option C is validated.

---

## Option B — Replace: GCN + BERT Instead of GVP-GNN

### Concept

Drop the GVP layers entirely. Replace them with CyberGFM's `GNNEmbedding`: a
standard multi-layer `GCNConv` initializes node embeddings from the protein residue
graph, and a BERT transformer contextualizes them via walk sequences. This is the
closest direct port of CyberGFM to the protein domain.

```
Input: Cα coordinates + backbone atoms + AA types
        │
        ▼  Feature extraction (invariant scalars only — no vectors)
    (N, D) scalar node features  +  edge_index
        │
        ▼  GCNConv (CyberGFM GNNEmbedding, e.g. 3 layers, 768-dim)
    Initial node embeddings  (N, 768)
        │
        ▼  Walk tokenizer  →  token sequences using GCN embeddings
        │
        ▼  BERT transformer (CyberGFM bert.py)
    Contextualized per-residue embeddings  (N, 768)
        │
        ▼  Linear projection
    Final per-residue embeddings  (N, 512)
```

### What changes in the codebase

| File | Change |
|---|---|
| `prostrencoder/models/gvp.py` | Kept but unused (GVPConv layers removed from encoder) |
| `prostrencoder/models/encoder.py` | Rewritten — GCNConv replaces GVPConv stack |
| `prostrencoder/models/walk_sampler.py` | New — adapted from CyberGFM `sampler.py` |
| `prostrencoder/models/walk_encoder.py` | New — CyberGFM `bert.py` or `gnn_bert.py` |
| `prostrencoder/data/features.py` | Modified — backbone frame vectors removed (no vector branch) |
| `prostrencoder/data/dataset.py` | Modified — `x_vec` field dropped |
| All tests for GVP, GVPConv, encoder | Rewritten |
| `configs/default.yaml` | `node_vector_in` removed; GCN depth/width added |

### Pros

- **Simplest conceptual alignment** with CyberGFM — almost a direct port.
- **Fewer moving parts**: one GCN + one transformer, no dual (scalar, vector) feature
  system.
- **Faster iteration**: standard GCN is well-supported by PyG with no special einsum
  logic.

### Cons

- **Loses SE(3)-equivariance**: `GCNConv` applied to 3D coordinate-derived features
  is not rotation-invariant unless all inputs are distances and angles (no raw xyz).
  Even with invariant inputs, the spectral aggregation is not equivariant to
  permutations of neighbors in 3D space.
- **Discards proven components**: the GVP layers (with their validated equivariance
  tests) are removed entirely; all their 3D geometric reasoning is lost.
- **Higher implementation risk**: rewriting the encoder and dataset invalidates the
  existing 48 tests. Significant test rewrite required.
- **Questionable benefit**: GCN without 3D awareness is weaker than GVP for protein
  structure; it essentially treats the protein graph as a flat network graph, losing
  the geometric information CyberGFM was never designed to capture.

### Recommended when

- The task is primarily **sequence-topology** (who is connected to whom) rather than
  **3D geometry** (how atoms are oriented in space).
- A fast proof-of-concept is needed with minimal geometric fidelity requirements.
- **Not recommended** as a primary approach for protein 3D structure tasks.

---

## Option C — Augment: Walk-Sequence Embedding Replaces RWSE *(Recommended for v1)*

### Concept

Keep the entire GVP-GNN architecture unchanged. Replace only the 16-dim RWSE
(random-walk landing probabilities) with a richer **walk-sequence embedding**
produced by a small transformer inspired by CyberGFM's tokenizer + BERT. The
walk embedding is concatenated to the scalar node features, exactly as RWSE is today.

```
Input: Cα coordinates + backbone atoms + AA types
        │
   ┌────┴──────────────────────────────────────────┐
   │ Existing feature extraction (unchanged)        │
   │   aa_one_hot  →  (N, 21)                       │
   │   torsion_angles  →  (N, 6)                    │
   └────┬──────────────────────────────────────────┘
        │
        ▼  WalkEncoder  (NEW — replaces compute_rwse)
   K random walks of length L per residue
   Each walk: sequence of AA-type tokens  [aa(i), aa(j1), ..., aa(jL)]
   4-layer, 128-dim transformer  →  mean-pool K walks
        │
   Walk embedding  (N, 64)
        │
   x_scalar = concat[one_hot(21), torsion(6), walk_emb(64)]  →  (N, 91)
        │
        ▼  GVP-GNN (3 × GVPConv, unchanged)  ← SE(3)-equivariant
   Per-residue embeddings  (N, 512)
```

### What changes in the codebase

| File | Change |
|---|---|
| `prostrencoder/models/walk_sampler.py` | New — static random walk sampler adapted from CyberGFM `sampler.py`; no temporal logic |
| `prostrencoder/models/walk_encoder.py` | New — small 4-layer, 128-dim transformer; `[CLS]` token + mean-pool over K walks → 64-dim |
| `prostrencoder/models/rwse.py` | Kept (backward compatibility); `WalkEncoder` is the new default |
| `scripts/preprocess.py` | Modified — `compute_rwse()` replaced by `WalkEncoder` inference at preprocessing time |
| `prostrencoder/data/dataset.py` | No change — `x_scalar` field name unchanged, just wider (91 vs 43) |
| `configs/default.yaml` | `node_scalar_in: 43 → 91`; add `walk_length: 20`, `num_walks: 8`, `walk_dim: 64`, `walk_layers: 4` |
| `tests/models/test_walk_encoder.py` | New — shape tests, walk sampling tests, embedding stability tests |
| All existing tests | **Unchanged** — GVP, GVPConv, encoder, parser, graph builder, objectives all stay green |

### How the CyberGFM technique adapts to proteins

CyberGFM's tokenizer converts graph nodes to categorical token IDs and feeds them
through a BERT transformer. For proteins, the same idea applies:

| Step | CyberGFM (network graph) | ProStrEncoder adaptation |
|---|---|---|
| Token vocabulary | IP addresses, process names | Amino acid types (21 tokens) |
| Walk source | Eulerian path or random walk | Static random walk on k-NN graph |
| Token at node *v* | `node_type[v]` | `aa_type[v]` (0–20) |
| Special tokens | `SOS`, `EOS`, `JUMP`, `MASK` | `CLS` (prepended to each walk) |
| Transformer | 12-layer BERT, 768-dim | 4-layer encoder, 128-dim |
| Output per node | masked token prediction | mean-pool K walks → 64-dim |

For each residue *i*, K independent random walks of length L are sampled. Each walk
becomes a token sequence `[CLS, aa(i), aa(j₁), aa(j₂), …, aa(j_L)]`. The transformer
processes each sequence and the K `[CLS]` hidden states are averaged to produce a
single 64-dim embedding that encodes *which amino acid types are topologically
reachable from residue i along different paths*.

### Pros

- **Zero regression risk**: all 48 existing tests continue to pass without modification.
- **Preserves SE(3)-equivariance**: the walk embedding feeds only the scalar channel;
  the vector channel (backbone frame) is untouched.
- **Strictly more informative than RWSE**: landing probabilities tell you *how likely*
  you are to return; walk sequences tell you *who you visit and what they are*. The
  transformer can learn arbitrary patterns over this richer signal.
- **Lightweight**: 4-layer, 128-dim transformer adds ~2.5 M parameters, not 85 M.
- **Preprocessing-time computation**: walk embeddings are stored in `.pt` files like
  RWSE is today; no walk sampling overhead during training.
- **Natural upgrade path**: once validated, the walk encoder can be promoted to the
  full BERT in Option A with minimal rework.
- **Lowest implementation effort**: ~2 new files, 2 modified files, config update.

### Cons

- **Walk embeddings are not equivariant**: they encode topology and AA-type context,
  not 3D geometry. This is intentional — 3D geometry is handled by GVP — but means
  the walk embedding cannot substitute for the vector features.
- **Preprocessing must be re-run**: existing `.pt` files store 16-dim RWSE; after
  the change all proteins must be re-preprocessed to store 64-dim walk embeddings.
- **Transformer on sequences of graph walks** adds a new module type that is less
  familiar than the GVP layers.

### Recommended when

- This is a **first integration** of CyberGFM technology.
- The existing GVP-GNN results should be preserved and extended, not replaced.
- Model size and training stability are a concern.
- A clear, low-risk validation of walk-based structural encoding is desired before
  committing to the heavier Option A.

---

## Comparison Summary

| | Option A | Option B | Option C |
|---|---|---|---|
| **Core idea** | GVP + Walk-BERT stacked | GCN + BERT replaces GVP | Walk embedding replaces RWSE |
| **SE(3)-equivariant** | ✅ Yes | ❌ No | ✅ Yes |
| **CyberGFM fidelity** | High (full BERT) | High (GCN + BERT) | Medium (small BERT, walk tokenizer) |
| **Existing tests broken** | Partial | Most | None |
| **New parameters** | ~85 M | ~85 M | ~2.5 M |
| **Implementation effort** | High | High | Low |
| **Preprocessing re-run** | Yes | Yes | Yes |
| **Recommended for v1** | ❌ | ❌ | ✅ |
| **Recommended for v2** | ✅ | — | — |

---

## Upgrade Path

```
v1  ──►  Option C  (walk embedding replaces RWSE)
              │
              │  Validate: do walk embeddings improve downstream task accuracy?
              │  Yes → proceed   No → tune K, L, transformer depth
              ▼
v2  ──►  Option A  (add full walk-sequence BERT after GVP-GNN)
              │
              │  Validate: does long-range transformer context add further lift?
              ▼
v3  ──►  Option A + temporal walks, pre-training on PDB-scale data
```

Option B is not on the primary upgrade path because discarding SE(3)-equivariance
is a structural regression for protein tasks, not an upgrade.

---

## References

- Jing, B. et al. (2021). **Learning from Protein Structure with Geometric Vector
  Perceptrons.** ICLR 2022. [arXiv:2009.01411](https://arxiv.org/abs/2009.01411)
- Rampášek, L. et al. (2022). **Recipe for a General, Powerful, Scalable Graph
  Transformer.** NeurIPS 2022. [arXiv:2205.12454](https://arxiv.org/abs/2205.12454)
- CyberGFM repository. [github.com/cybermonic/CyberGFM](https://github.com/cybermonic/CyberGFM)
