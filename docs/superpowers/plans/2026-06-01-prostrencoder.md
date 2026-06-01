# ProStrEncoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a foundational protein 3D structure encoder that produces per-residue embeddings from atomic coordinates using a GNN with Random Walk Structural Encoding (RWSE).

**Architecture:** Protein structures are parsed from PDB/mmCIF files and converted to directed k-NN residue graphs (k=30, Cα–Cα Euclidean distance). Node features combine amino acid one-hot encoding, backbone torsion angles (sin/cos), and RWSE positional encodings (16 steps); edge features combine RBF-encoded distances and unit direction vectors. Three GVP-GNN (Geometric Vector Perceptron) layers with equivariant message passing produce 512D per-residue embeddings, pre-trained via masked residue type prediction.

**Tech Stack:** Python 3.10+, PyTorch 2.x, PyTorch Geometric, torch-scatter, BioPython, NumPy, SciPy, PyYAML, pytest

---

## File Map

```
ProStrEncoder/
├── prostrencoder/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── parser.py          # PDB/mmCIF → residue dict
│   │   ├── graph_builder.py   # residue dict → k-NN graph (edge_index, edge_dist)
│   │   ├── features.py        # node/edge feature tensors
│   │   └── dataset.py         # PyG ProteinDataset (loads preprocessed .pt files)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── rwse.py            # random walk structural encoding (scipy, CPU)
│   │   ├── gvp.py             # GVP layer + GVPConv message-passing layer
│   │   ├── gnn.py             # stack of GVPConv layers
│   │   └── encoder.py         # full ProStrEncoder model
│   └── training/
│       ├── __init__.py
│       ├── objectives.py      # masked residue prediction loss
│       └── trainer.py         # training loop with AdamW + cosine LR
├── scripts/
│   ├── preprocess.py          # batch PDB → .pt PyG Data conversion
│   └── train.py               # training entry point
├── configs/
│   └── default.yaml           # hyperparameters
├── tests/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── test_parser.py
│   │   ├── test_graph_builder.py
│   │   ├── test_features.py
│   │   └── test_dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── test_rwse.py
│   │   ├── test_gvp.py
│   │   └── test_encoder.py
│   └── training/
│       ├── __init__.py
│       └── test_objectives.py
├── setup.py
└── requirements.txt
```

**Feature dimensions (baked in throughout):**

| Feature | Dim | Location |
|---------|-----|----------|
| AA one-hot | 21 | node scalar |
| Backbone torsion angles (sin/cos φ,ψ,ω) | 6 | node scalar |
| RWSE (16-step landing probabilities) | 16 | node scalar |
| **Total node scalars** | **43** | |
| Backbone frame (N→CA, CA→C, cross) | 3 vectors × 3D | node vector (N,3,3) |
| RBF distances (d_min=0, d_max=20Å, 16 bins) | 16 | edge scalar |
| src→dst unit direction | 1 vector × 3D | edge vector (E,1,3) |
| GVP hidden scalars | 128 | hidden |
| GVP hidden vectors | 16 | hidden |
| Output embedding | 512 | per residue |

---

### Task 1: Project Scaffold

**Files:**
- Create: `setup.py`
- Create: `requirements.txt`
- Create: `prostrencoder/__init__.py` and all `__init__.py` files
- Create: `configs/default.yaml`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p prostrencoder/{data,models,training}
mkdir -p tests/{data,models,training}
mkdir -p scripts configs
touch prostrencoder/__init__.py
touch prostrencoder/data/__init__.py
touch prostrencoder/models/__init__.py
touch prostrencoder/training/__init__.py
touch tests/__init__.py
touch tests/data/__init__.py
touch tests/models/__init__.py
touch tests/training/__init__.py
```

- [ ] **Step 2: Write `requirements.txt`**

```
torch>=2.1.0
torch-geometric>=2.4.0
torch-scatter>=2.1.0
biopython>=1.81
numpy>=1.24.0
scipy>=1.11.0
pyyaml>=6.0
pytest>=7.4.0
pytest-cov>=4.1.0
tqdm>=4.66.0
```

- [ ] **Step 3: Write `setup.py`**

```python
from setuptools import setup, find_packages

setup(
    name="prostrencoder",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torch-geometric>=2.4.0",
        "torch-scatter>=2.1.0",
        "biopython>=1.81",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
)
```

- [ ] **Step 4: Write `configs/default.yaml`**

```yaml
data:
  pdb_dir: data/pdb
  processed_dir: data/processed
  k_neighbors: 30
  rwse_steps: 16

model:
  num_layers: 3
  node_scalar_in: 43      # 21 AA + 6 torsion + 16 RWSE
  node_vector_in: 3       # backbone frame vectors
  edge_scalar_in: 16      # RBF distances
  edge_vector_in: 1       # direction vector
  hidden_scalar: 128
  hidden_vector: 16
  output_dim: 512
  dropout: 0.1

training:
  batch_size: 32
  max_epochs: 100
  lr: 1e-4
  weight_decay: 0.01
  mask_rate: 0.15
  grad_clip: 1.0
  checkpoint_dir: checkpoints
  log_every: 50
```

- [ ] **Step 5: Install package in editable mode**

```bash
pip install -e .
```

Expected: `Successfully installed prostrencoder-0.1.0`

- [ ] **Step 6: Verify imports**

```bash
python -c "import prostrencoder; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git init
git add setup.py requirements.txt prostrencoder/ tests/ scripts/ configs/ docs/
git commit -m "feat: project scaffold for ProStrEncoder"
```

---

### Task 2: PDB/mmCIF Parser

**Files:**
- Create: `prostrencoder/data/parser.py`
- Create: `tests/data/test_parser.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_parser.py`:

```python
import io
import tempfile
import os
import numpy as np
import pytest
from prostrencoder.data.parser import parse_structure, AA_TO_IDX

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
def pdb_file(tmp_path):
    p = tmp_path / "test.pdb"
    p.write_text(MINIMAL_PDB)
    return str(p)

def test_parse_structure_returns_expected_keys(pdb_file):
    result = parse_structure(pdb_file)
    assert set(result.keys()) == {"seq_idx", "ca_coords", "backbone_coords"}

def test_parse_structure_residue_count(pdb_file):
    result = parse_structure(pdb_file)
    assert result["seq_idx"].shape == (3,)
    assert result["ca_coords"].shape == (3, 3)
    assert result["backbone_coords"].shape == (3, 4, 3)

def test_parse_structure_aa_types(pdb_file):
    result = parse_structure(pdb_file)
    assert result["seq_idx"][0] == AA_TO_IDX["ALA"]
    assert result["seq_idx"][1] == AA_TO_IDX["GLY"]
    assert result["seq_idx"][2] == AA_TO_IDX["LEU"]

def test_parse_structure_ca_coords_are_finite(pdb_file):
    result = parse_structure(pdb_file)
    assert np.all(np.isfinite(result["ca_coords"]))

def test_parse_unknown_residue_gets_unk_index(tmp_path):
    pdb = """\
ATOM      1  N   XYZ A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  XYZ A   1      11.526  10.000  10.000  1.00  0.00           C
ATOM      3  C   XYZ A   1      12.000  11.400  10.000  1.00  0.00           C
ATOM      4  O   XYZ A   1      11.200  12.200  10.000  1.00  0.00           O
END
"""
    # XYZ is not a standard AA, but BioPython may not list it via is_aa
    # This test verifies that unrecognized residues map to UNK index (20)
    p = tmp_path / "unk.pdb"
    p.write_text(pdb)
    result = parse_structure(str(p))
    # If BioPython filters it out, length is 0; if kept, index is 20
    if len(result["seq_idx"]) > 0:
        assert result["seq_idx"][0] == 20
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/data/test_parser.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `parser.py` does not exist yet.

- [ ] **Step 3: Write `prostrencoder/data/parser.py`**

```python
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa

AA_CODES = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL", "UNK",
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_CODES)}


def parse_structure(filepath: str) -> dict:
    """
    Parse a PDB or mmCIF file and return per-residue arrays for all standard
    amino acid residues found in the first model, across all chains.

    Returns a dict with:
        seq_idx         : np.ndarray (N,) int64   — AA index (20 = UNK)
        ca_coords       : np.ndarray (N, 3) float32 — Cα coordinates (Å)
        backbone_coords : np.ndarray (N, 4, 3) float32 — N, CA, C, O coords
    """
    if filepath.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("prot", filepath)
    model = next(structure.get_models())

    seq_idx_list, ca_list, bb_list = [], [], []

    for chain in model:
        for res in chain:
            if not is_aa(res, standard=True):
                continue

            resname = res.get_resname().strip()
            seq_idx_list.append(AA_TO_IDX.get(resname, 20))

            if "CA" in res:
                ca_list.append(res["CA"].get_vector().get_array())
            else:
                ca_list.append(np.full(3, np.nan, dtype=np.float32))

            bb = []
            for atom_name in ("N", "CA", "C", "O"):
                if atom_name in res:
                    bb.append(res[atom_name].get_vector().get_array())
                else:
                    bb.append(np.full(3, np.nan, dtype=np.float32))
            bb_list.append(bb)

    return {
        "seq_idx": np.array(seq_idx_list, dtype=np.int64),
        "ca_coords": np.array(ca_list, dtype=np.float32),
        "backbone_coords": np.array(bb_list, dtype=np.float32),
    }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/data/test_parser.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add prostrencoder/data/parser.py tests/data/test_parser.py
git commit -m "feat: PDB/mmCIF parser returning per-residue arrays"
```

---

### Task 3: Graph Builder

**Files:**
- Create: `prostrencoder/data/graph_builder.py`
- Create: `tests/data/test_graph_builder.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_graph_builder.py`:

```python
import numpy as np
import pytest
from prostrencoder.data.graph_builder import build_knn_graph

def linear_chain_coords(n, spacing=3.8):
    """n residues spaced spacing Å apart along x-axis."""
    return np.stack([np.arange(n) * spacing,
                     np.zeros(n),
                     np.zeros(n)], axis=1).astype(np.float32)

def test_knn_graph_shape_small():
    coords = linear_chain_coords(5)
    edge_index, edge_dist = build_knn_graph(coords, k=2)
    assert edge_index.shape[0] == 2
    assert edge_dist.shape[0] == edge_index.shape[1]
    assert edge_index.dtype == np.int64
    assert edge_dist.dtype == np.float32

def test_knn_graph_no_self_loops():
    coords = linear_chain_coords(10)
    edge_index, _ = build_knn_graph(coords, k=4)
    src, dst = edge_index
    assert not np.any(src == dst), "Graph must not contain self-loops"

def test_knn_graph_k_neighbors_per_node():
    n, k = 20, 5
    coords = linear_chain_coords(n)
    edge_index, _ = build_knn_graph(coords, k=k)
    src = edge_index[0]
    # Each non-boundary node should have exactly k outgoing edges;
    # boundary nodes may have fewer if k > (n-1).
    counts = np.bincount(src, minlength=n)
    assert np.all(counts <= k)
    # Interior nodes get exactly k neighbors
    assert np.all(counts[1:-1] == k)

def test_knn_graph_distances_are_positive():
    coords = linear_chain_coords(10)
    _, edge_dist = build_knn_graph(coords, k=3)
    assert np.all(edge_dist > 0)

def test_knn_graph_k_larger_than_n_does_not_crash():
    coords = linear_chain_coords(3)
    edge_index, edge_dist = build_knn_graph(coords, k=100)
    # Should not crash; each node gets at most n-1 = 2 neighbors
    assert edge_index.shape[1] == 3 * 2  # 3 nodes × 2 edges each (bidirectional)

def test_knn_graph_distances_match_euclidean():
    coords = linear_chain_coords(5, spacing=3.8)
    edge_index, edge_dist = build_knn_graph(coords, k=1)
    src, dst = edge_index
    for i in range(len(src)):
        expected = np.linalg.norm(coords[dst[i]] - coords[src[i]])
        assert abs(edge_dist[i] - expected) < 1e-4
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/data/test_graph_builder.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `prostrencoder/data/graph_builder.py`**

```python
import numpy as np
from scipy.spatial import cKDTree


def build_knn_graph(ca_coords: np.ndarray, k: int = 30):
    """
    Build a directed k-NN graph from Cα coordinates.

    Residues with NaN coordinates are placed at a sentinel far-field
    location so they form no meaningful edges.

    Args:
        ca_coords : (N, 3) float32 array of Cα positions.
        k         : number of nearest neighbours per node (excluding self).

    Returns:
        edge_index : (2, E) int64 — [src_indices, dst_indices]
        edge_dist  : (E,) float32 — Euclidean distances in Å
    """
    n = len(ca_coords)
    coords = ca_coords.copy()
    nan_mask = np.isnan(coords).any(axis=1)
    coords[nan_mask] = 1e9  # sentinel: far from everything

    k_actual = min(k + 1, n)  # +1 because query includes self
    tree = cKDTree(coords)
    dists, indices = tree.query(coords, k=k_actual)

    src_list, dst_list, dist_list = [], [], []
    for i in range(n):
        for j, d in zip(indices[i], dists[i]):
            if j != i:
                src_list.append(i)
                dst_list.append(int(j))
                dist_list.append(d)

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_dist = np.array(dist_list, dtype=np.float32)
    return edge_index, edge_dist
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/data/test_graph_builder.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add prostrencoder/data/graph_builder.py tests/data/test_graph_builder.py
git commit -m "feat: k-NN graph builder from Cα coordinates"
```

---

### Task 4: Node and Edge Feature Computation

**Files:**
- Create: `prostrencoder/data/features.py`
- Create: `tests/data/test_features.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_features.py`:

```python
import numpy as np
import pytest
from prostrencoder.data.features import (
    aa_one_hot,
    compute_backbone_frame,
    compute_torsion_angles,
    rbf_encoding,
    compute_edge_directions,
)

N = 10

def make_linear_backbone(n=N, spacing=3.8):
    """Linear chain of residues: N, CA, C, O placed regularly."""
    bb = np.zeros((n, 4, 3), dtype=np.float32)
    for i in range(n):
        x = i * spacing
        bb[i, 0] = [x - 1.5, 0.0, 0.0]   # N
        bb[i, 1] = [x,       0.0, 0.0]   # CA
        bb[i, 2] = [x + 1.2, 0.4, 0.0]   # C
        bb[i, 3] = [x + 1.2, 1.6, 0.0]   # O
    return bb

def test_aa_one_hot_shape():
    seq_idx = np.array([0, 1, 20], dtype=np.int64)
    out = aa_one_hot(seq_idx)
    assert out.shape == (3, 21)
    assert out.dtype == np.float32

def test_aa_one_hot_sum_one():
    seq_idx = np.arange(21, dtype=np.int64)
    out = aa_one_hot(seq_idx)
    np.testing.assert_array_equal(out.sum(axis=1), np.ones(21))

def test_backbone_frame_shape():
    bb = make_linear_backbone()
    frame = compute_backbone_frame(bb)
    assert frame.shape == (N, 3, 3)
    assert frame.dtype == np.float32

def test_backbone_frame_orthonormal():
    bb = make_linear_backbone()
    frame = compute_backbone_frame(bb)
    for i in range(N):
        v1, v2, v3 = frame[i]
        np.testing.assert_allclose(np.linalg.norm(v1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(v2), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(v3), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.dot(v1, v2), 0.0, atol=1e-5)

def test_torsion_angles_shape():
    bb = make_linear_backbone()
    angles = compute_torsion_angles(bb)
    assert angles.shape == (N, 6)
    assert angles.dtype == np.float32

def test_torsion_angles_sin_cos_range():
    bb = make_linear_backbone()
    angles = compute_torsion_angles(bb)
    assert np.all(angles >= -1.0 - 1e-6)
    assert np.all(angles <=  1.0 + 1e-6)

def test_rbf_encoding_shape():
    dists = np.array([0.0, 5.0, 10.0, 20.0], dtype=np.float32)
    rbf = rbf_encoding(dists, num_rbf=16)
    assert rbf.shape == (4, 16)
    assert rbf.dtype == np.float32

def test_rbf_encoding_peaks_at_center():
    centers = np.linspace(0, 20, 16)
    for c in centers:
        dists = np.array([c], dtype=np.float32)
        rbf = rbf_encoding(dists, num_rbf=16)
        peak_idx = np.argmax(rbf[0])
        peak_center = centers[peak_idx]
        assert abs(peak_center - c) < 2.0

def test_edge_directions_unit_length():
    ca = np.array([[0,0,0],[3,4,0],[0,3,4]], dtype=np.float32)
    edge_index = np.array([[0,1],[1,2]], dtype=np.int64)
    edge_dist = np.array([5.0, 5.0], dtype=np.float32)
    dirs = compute_edge_directions(ca, edge_index, edge_dist)
    assert dirs.shape == (2, 3)
    norms = np.linalg.norm(dirs, axis=1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/data/test_features.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `prostrencoder/data/features.py`**

```python
import numpy as np


AA_CODES = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL", "UNK",
]


def aa_one_hot(seq_idx: np.ndarray) -> np.ndarray:
    """One-hot encode AA indices. Returns (N, 21) float32."""
    n = len(seq_idx)
    out = np.zeros((n, 21), dtype=np.float32)
    out[np.arange(n), seq_idx.clip(0, 20)] = 1.0
    return out


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8) if norm > 1e-8 else np.zeros_like(v)


def compute_backbone_frame(backbone_coords: np.ndarray) -> np.ndarray:
    """
    Compute a local orthonormal frame per residue from backbone atoms.

    backbone_coords : (N, 4, 3) — columns are N, CA, C, O.

    Returns (N, 3, 3) where frame[i] = [v1, v2, v3], three mutually
    orthogonal unit vectors describing the local backbone orientation.
      v1 = N → CA (normalized)
      v2 = component of CA → C perpendicular to v1 (Gram-Schmidt)
      v3 = v1 × v2
    """
    N_coords  = backbone_coords[:, 0]   # (N, 3)
    CA_coords = backbone_coords[:, 1]
    C_coords  = backbone_coords[:, 2]

    n = len(CA_coords)
    frame = np.zeros((n, 3, 3), dtype=np.float32)

    for i in range(n):
        v1 = _safe_normalize(CA_coords[i] - N_coords[i])
        c_vec = C_coords[i] - CA_coords[i]
        v2 = _safe_normalize(c_vec - np.dot(c_vec, v1) * v1)
        v3 = np.cross(v1, v2).astype(np.float32)
        v3 = _safe_normalize(v3)
        frame[i] = np.stack([v1, v2, v3])

    return frame


def _dihedral(p0, p1, p2, p3) -> float:
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-6 or n2_norm < 1e-6:
        return 0.0
    n1 /= n1_norm
    n2 /= n2_norm
    b2_hat = b2 / (np.linalg.norm(b2) + 1e-8)
    m1 = np.cross(n1, b2_hat)
    return float(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))


def compute_torsion_angles(backbone_coords: np.ndarray) -> np.ndarray:
    """
    Compute backbone torsion angles φ, ψ, ω as sin/cos pairs.

    backbone_coords : (N, 4, 3) — N, CA, C, O per residue.

    Returns (N, 6) float32:
      cols 0,1 = sin(φ), cos(φ)
      cols 2,3 = sin(ψ), cos(ψ)
      cols 4,5 = sin(ω), cos(ω)

    Terminal residues get zeros for undefined angles.
    """
    n = len(backbone_coords)
    N_idx, CA_idx, C_idx = 0, 1, 2
    out = np.zeros((n, 6), dtype=np.float32)

    for i in range(n):
        if i > 0:
            phi = _dihedral(
                backbone_coords[i - 1, C_idx],
                backbone_coords[i, N_idx],
                backbone_coords[i, CA_idx],
                backbone_coords[i, C_idx],
            )
            out[i, 0] = np.sin(phi)
            out[i, 1] = np.cos(phi)

        if i < n - 1:
            psi = _dihedral(
                backbone_coords[i, N_idx],
                backbone_coords[i, CA_idx],
                backbone_coords[i, C_idx],
                backbone_coords[i + 1, N_idx],
            )
            out[i, 2] = np.sin(psi)
            out[i, 3] = np.cos(psi)

        if i > 0:
            omega = _dihedral(
                backbone_coords[i - 1, CA_idx],
                backbone_coords[i - 1, C_idx],
                backbone_coords[i, N_idx],
                backbone_coords[i, CA_idx],
            )
            out[i, 4] = np.sin(omega)
            out[i, 5] = np.cos(omega)

    return out


def rbf_encoding(distances: np.ndarray, num_rbf: int = 16,
                 d_min: float = 0.0, d_max: float = 20.0) -> np.ndarray:
    """
    Gaussian RBF encoding of edge distances.

    distances : (E,) float32
    Returns   : (E, num_rbf) float32
    """
    centers = np.linspace(d_min, d_max, num_rbf, dtype=np.float32)
    sigma = (d_max - d_min) / (num_rbf - 1)
    return np.exp(-((distances[:, None] - centers[None, :]) ** 2) / (2 * sigma ** 2))


def compute_edge_directions(ca_coords: np.ndarray,
                             edge_index: np.ndarray,
                             edge_dist: np.ndarray) -> np.ndarray:
    """
    Unit direction vectors from source Cα to destination Cα.

    Returns (E, 3) float32.
    """
    src, dst = edge_index[0], edge_index[1]
    diff = ca_coords[dst] - ca_coords[src]
    return (diff / (edge_dist[:, None] + 1e-8)).astype(np.float32)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/data/test_features.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add prostrencoder/data/features.py tests/data/test_features.py
git commit -m "feat: node/edge feature computation (AA one-hot, torsion, RBF, backbone frame)"
```

---

### Task 5: RWSE Computation

**Files:**
- Create: `prostrencoder/models/rwse.py`
- Create: `tests/models/test_rwse.py`

- [ ] **Step 1: Write the failing test**

`tests/models/test_rwse.py`:

```python
import numpy as np
import pytest
from prostrencoder.models.rwse import compute_rwse

def ring_graph(n):
    """Undirected ring graph: node i connects to (i-1) and (i+1) mod n."""
    src = list(range(n)) + list(range(n))
    dst = [(i + 1) % n for i in range(n)] + [(i - 1) % n for i in range(n)]
    return np.array([src, dst], dtype=np.int64)

def test_rwse_output_shape():
    edge_index = ring_graph(6)
    rwse = compute_rwse(edge_index, num_nodes=6, walk_length=8)
    assert rwse.shape == (6, 8)
    assert rwse.dtype == np.float32

def test_rwse_values_in_zero_one():
    edge_index = ring_graph(8)
    rwse = compute_rwse(edge_index, num_nodes=8, walk_length=8)
    assert np.all(rwse >= 0.0 - 1e-6)
    assert np.all(rwse <= 1.0 + 1e-6)

def test_rwse_symmetric_on_ring():
    """All nodes in a ring are equivalent; their RWSE should be identical."""
    n = 6
    edge_index = ring_graph(n)
    rwse = compute_rwse(edge_index, num_nodes=n, walk_length=8)
    for i in range(1, n):
        np.testing.assert_allclose(rwse[0], rwse[i], atol=1e-5)

def test_rwse_step1_is_zero_on_ring():
    """In a ring, a 1-step walk never returns to start (no self-loops)."""
    edge_index = ring_graph(6)
    rwse = compute_rwse(edge_index, num_nodes=6, walk_length=4)
    np.testing.assert_allclose(rwse[:, 0], np.zeros(6), atol=1e-6)

def test_rwse_step2_nonzero_on_ring():
    """In a 4-ring, a 2-step walk has 50% chance of returning."""
    edge_index = ring_graph(4)
    rwse = compute_rwse(edge_index, num_nodes=4, walk_length=4)
    np.testing.assert_allclose(rwse[:, 1], np.full(4, 0.5), atol=1e-5)

def test_rwse_disconnected_node():
    """An isolated node has RWSE = 0 for all steps."""
    # 3-node graph: 0-1-2 chain, node 3 is isolated
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    rwse = compute_rwse(edge_index, num_nodes=4, walk_length=4)
    np.testing.assert_allclose(rwse[3], np.zeros(4), atol=1e-6)
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/models/test_rwse.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `prostrencoder/models/rwse.py`**

```python
import numpy as np
from scipy.sparse import csr_matrix, eye as speye


def compute_rwse(edge_index: np.ndarray, num_nodes: int,
                 walk_length: int = 16) -> np.ndarray:
    """
    Compute Random Walk Structural Encoding (RWSE).

    RWSE[i, k] = probability of returning to node i after exactly k+1 steps,
    starting from node i, using the row-stochastic random walk matrix
    P = D^{-1} A (A = adjacency, D = out-degree diagonal).

    Isolated nodes (degree 0) get RWSE = 0.

    Args:
        edge_index  : (2, E) int64 — directed edge list [src, dst]
        num_nodes   : N
        walk_length : K — number of steps to compute

    Returns:
        rwse : (N, K) float32
    """
    N = num_nodes
    src = edge_index[0]
    dst = edge_index[1]

    data = np.ones(len(src), dtype=np.float64)
    A = csr_matrix((data, (src, dst)), shape=(N, N))

    # Row-stochastic P = D^{-1} A
    deg = np.array(A.sum(axis=1), dtype=np.float64).flatten()
    inv_deg = np.where(deg > 0, 1.0 / deg, 0.0)
    D_inv = csr_matrix(
        (inv_deg, (np.arange(N), np.arange(N))), shape=(N, N)
    )
    P = D_inv @ A

    rwse = np.zeros((N, walk_length), dtype=np.float32)
    P_k = speye(N, format="csr", dtype=np.float64)  # P^0 = I

    for k in range(walk_length):
        P_k = P_k @ P
        rwse[:, k] = P_k.diagonal().astype(np.float32)

    return rwse
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/models/test_rwse.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add prostrencoder/models/rwse.py tests/models/test_rwse.py
git commit -m "feat: random walk structural encoding (RWSE) via scipy sparse"
```

---

### Task 6: Preprocessing Script and PyG Dataset

**Files:**
- Create: `scripts/preprocess.py`
- Create: `prostrencoder/data/dataset.py`
- Create: `tests/data/test_dataset.py`

- [ ] **Step 1: Write the failing dataset test**

`tests/data/test_dataset.py`:

```python
import os
import tempfile
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
    """Create one preprocessed .pt file and return the directory."""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text(MINIMAL_PDB)

    # Build the Data object manually (mimics preprocess.py output)
    from prostrencoder.data.parser import parse_structure
    from prostrencoder.data.graph_builder import build_knn_graph
    from prostrencoder.data.features import (
        aa_one_hot, compute_backbone_frame, compute_torsion_angles,
        rbf_encoding, compute_edge_directions,
    )
    from prostrencoder.models.rwse import compute_rwse

    parsed = parse_structure(str(pdb_path))
    edge_index, edge_dist = build_knn_graph(parsed["ca_coords"], k=30)
    onehot = aa_one_hot(parsed["seq_idx"])
    torsion = compute_torsion_angles(parsed["backbone_coords"])
    frame = compute_backbone_frame(parsed["backbone_coords"])
    rbf = rbf_encoding(edge_dist)
    dirs = compute_edge_directions(parsed["ca_coords"], edge_index, edge_dist)
    rwse = compute_rwse(edge_index, num_nodes=len(parsed["seq_idx"]), walk_length=16)

    x_scalar = np.concatenate([onehot, torsion, rwse], axis=1)  # (N, 43)
    x_vec = frame[:, np.newaxis, :, :]  # wrong shape — fix: (N, 3, 3) is already right
    # frame: (N, 3, 3) — already 3 vectors per node
    e_scalar = rbf                      # (E, 16)
    e_vec = dirs[:, np.newaxis, :]      # (E, 1, 3)

    data = Data(
        seq_idx=torch.from_numpy(parsed["seq_idx"]),
        x_scalar=torch.from_numpy(x_scalar),
        x_vec=torch.from_numpy(frame),           # (N, 3, 3)
        edge_index=torch.from_numpy(edge_index),
        edge_scalar=torch.from_numpy(e_scalar),  # (E, 16)
        edge_vec=torch.from_numpy(e_vec),         # (E, 1, 3)
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
    assert item.x_scalar.shape == (N, 43)
    assert item.x_vec.shape == (N, 3, 3)
    assert item.edge_scalar.shape[1] == 16
    assert item.edge_vec.shape[1:] == (1, 3)
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/data/test_dataset.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `prostrencoder/data/dataset.py`**

```python
import os
import torch
from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    """
    Loads preprocessed PyG Data objects from a directory of `.pt` files.
    Each file was created by scripts/preprocess.py.
    """

    def __init__(self, processed_dir: str):
        self.files = sorted([
            os.path.join(processed_dir, f)
            for f in os.listdir(processed_dir)
            if f.endswith(".pt")
        ])
        if not self.files:
            raise RuntimeError(f"No .pt files found in {processed_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=False)
```

- [ ] **Step 4: Write `scripts/preprocess.py`**

```python
"""
Convert a directory of PDB files to preprocessed PyG Data objects.

Usage:
    python scripts/preprocess.py --pdb_dir data/pdb --out_dir data/processed --k 30 --rwse_steps 16
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
from prostrencoder.models.rwse import compute_rwse


def preprocess_one(pdb_path: str, k: int, rwse_steps: int) -> Data:
    parsed = parse_structure(pdb_path)
    if len(parsed["seq_idx"]) < 4:
        raise ValueError(f"Too few residues in {pdb_path}")

    edge_index, edge_dist = build_knn_graph(parsed["ca_coords"], k=k)

    onehot  = aa_one_hot(parsed["seq_idx"])                          # (N, 21)
    torsion = compute_torsion_angles(parsed["backbone_coords"])       # (N, 6)
    frame   = compute_backbone_frame(parsed["backbone_coords"])       # (N, 3, 3)
    rbf     = rbf_encoding(edge_dist)                                 # (E, 16)
    dirs    = compute_edge_directions(parsed["ca_coords"], edge_index, edge_dist)  # (E, 3)
    rwse    = compute_rwse(edge_index, num_nodes=len(parsed["seq_idx"]),
                           walk_length=rwse_steps)                    # (N, rwse_steps)

    x_scalar  = np.concatenate([onehot, torsion, rwse], axis=1).astype(np.float32)  # (N, 43)
    e_vec     = dirs[:, np.newaxis, :].astype(np.float32)            # (E, 1, 3)

    return Data(
        seq_idx=torch.from_numpy(parsed["seq_idx"]),
        x_scalar=torch.from_numpy(x_scalar),
        x_vec=torch.from_numpy(frame),                # (N, 3, 3)
        edge_index=torch.from_numpy(edge_index),
        edge_scalar=torch.from_numpy(rbf),            # (E, 16)
        edge_vec=torch.from_numpy(e_vec),             # (E, 1, 3)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir",    required=True)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--k",          type=int, default=30)
    parser.add_argument("--rwse_steps", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pdb_files = [f for f in os.listdir(args.pdb_dir) if f.endswith((".pdb", ".cif"))]

    skipped = 0
    for fname in tqdm(pdb_files, desc="Preprocessing"):
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(args.out_dir, stem + ".pt")
        if os.path.exists(out_path):
            continue
        try:
            data = preprocess_one(os.path.join(args.pdb_dir, fname), args.k, args.rwse_steps)
            torch.save(data, out_path)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            skipped += 1

    print(f"Done. Skipped {skipped}/{len(pdb_files)} files.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/data/test_dataset.py -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add prostrencoder/data/dataset.py scripts/preprocess.py tests/data/test_dataset.py
git commit -m "feat: PyG ProteinDataset and preprocessing script"
```

---

### Task 7: GVP Layer

**Files:**
- Create: `prostrencoder/models/gvp.py`
- Create: `tests/models/test_gvp.py`

- [ ] **Step 1: Write the failing test**

`tests/models/test_gvp.py`:

```python
import torch
import pytest
from prostrencoder.models.gvp import GVP, GVPConv

def rand_node_features(n, s_dim, v_dim):
    return torch.randn(n, s_dim), torch.randn(n, v_dim, 3)

def rand_edge_features(e, s_dim, v_dim):
    return torch.randn(e, s_dim), torch.randn(e, v_dim, 3)

# ── GVP tests ─────────────────────────────────────────────────────────────────

def test_gvp_output_shapes():
    gvp = GVP(in_dims=(27, 3), out_dims=(64, 8))
    s, V = rand_node_features(10, 27, 3)
    s_out, V_out = gvp(s, V)
    assert s_out.shape == (10, 64)
    assert V_out.shape == (10, 8, 3)

def test_gvp_no_vector_out():
    gvp = GVP(in_dims=(16, 4), out_dims=(32, 0))
    s, V = rand_node_features(5, 16, 4)
    s_out, V_out = gvp(s, V)
    assert s_out.shape == (5, 32)
    assert V_out.shape == (5, 0, 3)

def test_gvp_scalar_only_input():
    gvp = GVP(in_dims=(16, 0), out_dims=(32, 4))
    s = torch.randn(5, 16)
    V = torch.zeros(5, 0, 3)
    s_out, V_out = gvp(s, V)
    assert s_out.shape == (5, 32)
    assert V_out.shape == (5, 4, 3)

def test_gvp_equivariant_vectors():
    """Rotating input vectors by R should rotate output vectors by R."""
    gvp = GVP(in_dims=(8, 4), out_dims=(8, 4))
    gvp.eval()
    s = torch.randn(3, 8)
    V = torch.randn(3, 4, 3)

    # Random rotation matrix
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    V_rot = (Q @ V.transpose(-1, -2)).transpose(-1, -2)

    with torch.no_grad():
        s_out, V_out = gvp(s, V)
        s_out_rot, V_out_rot = gvp(s, V_rot)

    torch.testing.assert_close(s_out, s_out_rot, atol=1e-5, rtol=1e-4)
    expected_V_out_rot = (Q @ V_out.transpose(-1, -2)).transpose(-1, -2)
    torch.testing.assert_close(V_out_rot, expected_V_out_rot, atol=1e-5, rtol=1e-4)

# ── GVPConv tests ──────────────────────────────────────────────────────────────

def make_chain_graph(n=6):
    """Bidirectional chain: 0↔1↔2↔…↔(n-1)."""
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    return torch.tensor([src, dst], dtype=torch.long)

def test_gvpconv_output_shapes():
    conv = GVPConv(
        node_in_dims=(43, 3),
        edge_in_dims=(16, 1),
        node_out_dims=(128, 16),
    )
    N, E = 10, 20
    node_s, node_v = rand_node_features(N, 43, 3)
    edge_s, edge_v = rand_edge_features(E, 16, 1)
    edge_index = make_chain_graph(N)[:, :E]
    out_s, out_v = conv(node_s, node_v, edge_index, edge_s, edge_v)
    assert out_s.shape == (N, 128)
    assert out_v.shape == (N, 16, 3)

def test_gvpconv_equivariant():
    """Rotating all 3D vectors by R should rotate output vectors by R."""
    conv = GVPConv(
        node_in_dims=(8, 2), edge_in_dims=(4, 1), node_out_dims=(8, 2),
    )
    conv.eval()
    N, E = 6, 8
    node_s, node_v = rand_node_features(N, 8, 2)
    edge_s, edge_v = rand_edge_features(E, 4, 1)
    edge_index = torch.randint(0, N, (2, E))

    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    node_v_rot = (Q @ node_v.transpose(-1, -2)).transpose(-1, -2)
    edge_v_rot = (Q @ edge_v.transpose(-1, -2)).transpose(-1, -2)

    with torch.no_grad():
        out_s, out_v = conv(node_s, node_v, edge_index, edge_s, edge_v)
        out_s_rot, out_v_rot = conv(node_s, node_v_rot, edge_index, edge_s, edge_v_rot)

    torch.testing.assert_close(out_s, out_s_rot, atol=1e-4, rtol=1e-3)
    expected = (Q @ out_v.transpose(-1, -2)).transpose(-1, -2)
    torch.testing.assert_close(out_v_rot, expected, atol=1e-4, rtol=1e-3)
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/models/test_gvp.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `prostrencoder/models/gvp.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GVP(nn.Module):
    """
    Geometric Vector Perceptron (Jing et al. 2021).

    Transforms (scalar_feats, vector_feats) → (scalar_out, vector_out)
    while preserving E(3) equivariance of the vector branch.

    in_dims  = (s_in, v_in)   where s_in is scalar dim, v_in is # of 3D vectors
    out_dims = (s_out, v_out)
    """

    def __init__(self, in_dims: tuple, out_dims: tuple,
                 scalar_act=nn.ReLU(), vector_gate: bool = True):
        super().__init__()
        s_in, v_in = in_dims
        s_out, v_out = out_dims

        self.v_in  = v_in
        self.v_out = v_out
        self.vector_gate = vector_gate

        # v_h: hidden vector dim used to squeeze vector info into scalars
        v_h = max(v_in, v_out)

        self.W_h = nn.Linear(v_in, v_h, bias=False) if v_in > 0 else None
        self.W_V = nn.Linear(v_in, v_out, bias=False) if v_in > 0 and v_out > 0 else None

        # scalar projection: s_in + v_h norms → s_out
        self.W_s = nn.Linear(s_in + (v_h if v_in > 0 else 0), s_out)

        if vector_gate and v_out > 0:
            self.gate_linear = nn.Linear(s_out, v_out)

        self.scalar_act = scalar_act

    @staticmethod
    def _vec_linear(W: nn.Linear, V: torch.Tensor) -> torch.Tensor:
        """Apply W to the vector-channel dimension of V (..., v_in, 3)."""
        return torch.einsum("...vi,oi->...vo", V, W.weight)

    def forward(self, s: torch.Tensor, V: torch.Tensor):
        """
        s : (..., s_in)
        V : (..., v_in, 3)
        Returns (s_out, V_out) with shapes (..., s_out) and (..., v_out, 3).
        """
        # ── Vector → scalar (via norms) ───────────────────────────────────────
        if self.v_in > 0 and self.W_h is not None:
            V_h = self._vec_linear(self.W_h, V)          # (..., v_h, 3)
            norms = torch.norm(V_h, dim=-1)               # (..., v_h)
            s_cat = torch.cat([s, norms], dim=-1)
        else:
            s_cat = s

        # ── Scalar branch ─────────────────────────────────────────────────────
        s_out = self.W_s(s_cat)
        if self.scalar_act is not None:
            s_out = self.scalar_act(s_out)

        # ── Vector branch ─────────────────────────────────────────────────────
        if self.v_out == 0 or self.v_in == 0 or self.W_V is None:
            V_out = V.new_zeros(*V.shape[:-2], self.v_out, 3)
        else:
            V_out = self._vec_linear(self.W_V, V)         # (..., v_out, 3)
            if self.vector_gate:
                gates = torch.sigmoid(self.gate_linear(s_out))  # (..., v_out)
                V_out = gates.unsqueeze(-1) * V_out

        return s_out, V_out


class GVPConv(nn.Module):
    """
    One round of GVP-based message passing.

    Messages are computed by concatenating source-node and edge features and
    passing them through two GVP layers.  Messages are sum-aggregated at
    destination nodes.  Node states are updated by concatenating the
    destination-node state with the aggregated messages and passing through
    two more GVP layers, with a residual connection on the scalar branch.
    """

    def __init__(self, node_in_dims: tuple, edge_in_dims: tuple,
                 node_out_dims: tuple, drop_rate: float = 0.1):
        super().__init__()
        s_n, v_n = node_in_dims
        s_e, v_e = edge_in_dims
        s_o, v_o = node_out_dims

        # Message network: (src_node ∥ edge) → message
        self.msg1 = GVP((s_n + s_e, v_n + v_e), (s_o, v_o))
        self.msg2 = GVP((s_o, v_o), (s_o, v_o), scalar_act=None, vector_gate=False)

        # Update network: (dst_node ∥ agg_message) → new node state
        self.upd1 = GVP((s_n + s_o, v_n + v_o), (s_o, v_o))
        self.upd2 = GVP((s_o, v_o), (s_o, v_o), scalar_act=None, vector_gate=False)

        self.norm_s = nn.LayerNorm(s_o)
        self.drop   = nn.Dropout(drop_rate)

        # Scalar residual projection when dims differ
        self.res_proj = nn.Linear(s_n, s_o, bias=False) if s_n != s_o else nn.Identity()

    def forward(self, node_s, node_v, edge_index, edge_s, edge_v):
        """
        node_s : (N, s_n)    node_v : (N, v_n, 3)
        edge_s : (E, s_e)    edge_v : (E, v_e, 3)
        edge_index : (2, E)
        Returns updated (node_s, node_v) with dims (s_o, v_o).
        """
        src, dst = edge_index[0], edge_index[1]
        N = node_s.shape[0]

        # ── Build messages ────────────────────────────────────────────────────
        m_s = torch.cat([node_s[src], edge_s], dim=-1)          # (E, s_n+s_e)
        m_v = torch.cat([node_v[src], edge_v], dim=-2)          # (E, v_n+v_e, 3)

        m_s, m_v = self.msg1(m_s, m_v)
        m_s, m_v = self.msg2(m_s, m_v)

        # ── Aggregate messages (sum) ──────────────────────────────────────────
        agg_s = node_s.new_zeros(N, m_s.shape[-1])
        agg_s.scatter_add_(0, dst.unsqueeze(-1).expand_as(m_s), m_s)

        agg_v = node_v.new_zeros(N, m_v.shape[-2], 3)
        dst_idx = dst.view(-1, 1, 1).expand_as(m_v)
        agg_v.scatter_add_(0, dst_idx, m_v)

        # ── Update node states ────────────────────────────────────────────────
        upd_s = torch.cat([node_s, agg_s], dim=-1)              # (N, s_n+s_o)
        upd_v = torch.cat([node_v, agg_v], dim=-2)              # (N, v_n+v_o, 3)

        new_s, new_v = self.upd1(upd_s, upd_v)
        new_s, new_v = self.upd2(new_s, new_v)

        new_s = self.norm_s(self.drop(new_s) + self.res_proj(node_s))

        return new_s, new_v
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/models/test_gvp.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add prostrencoder/models/gvp.py tests/models/test_gvp.py
git commit -m "feat: GVP layer and GVPConv message-passing with equivariance"
```

---

### Task 8: ProStrEncoder Model

**Files:**
- Create: `prostrencoder/models/encoder.py`
- Create: `tests/models/test_encoder.py`

- [ ] **Step 1: Write the failing test**

`tests/models/test_encoder.py`:

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
            x_scalar=torch.randn(N, 43),
            x_vec=torch.randn(N, 3, 3),
            edge_index=torch.randint(0, N, (2, E)),
            edge_scalar=torch.randn(E, 16),
            edge_vec=torch.randn(E, 1, 3),
        )
        data_list.append(data)
    return Batch.from_data_list(data_list)

CONFIG = {
    "num_layers": 2,
    "node_scalar_in": 43,
    "node_vector_in": 3,
    "edge_scalar_in": 16,
    "edge_vector_in": 1,
    "hidden_scalar": 64,
    "hidden_vector": 8,
    "output_dim": 128,
    "dropout": 0.0,
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
    """Embedding (invariant scalars) must not change when input vectors are rotated."""
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

    torch.testing.assert_close(emb, emb_rot, atol=1e-4, rtol=1e-3)

def test_encoder_parameter_count():
    model = ProStrEncoder(CONFIG)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 1_000, "Model seems too small"
    assert n_params < 100_000_000, "Model seems too large for config"
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/models/test_encoder.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `prostrencoder/models/encoder.py`**

```python
import torch
import torch.nn as nn
from prostrencoder.models.gvp import GVP, GVPConv


class ProStrEncoder(nn.Module):
    """
    GVP-GNN protein structure encoder.

    Input : PyG Batch / Data with fields
              x_scalar   (N, 43)    scalar node features
              x_vec      (N, 3, 3)  vector node features (backbone frame)
              edge_index (2, E)
              edge_scalar (E, 16)
              edge_vec    (E, 1, 3)

    Output: (N, output_dim) rotation-invariant per-residue embeddings.
    """

    def __init__(self, config: dict):
        super().__init__()
        s_n  = config["node_scalar_in"]   # 43
        v_n  = config["node_vector_in"]   # 3
        s_e  = config["edge_scalar_in"]   # 16
        v_e  = config["edge_vector_in"]   # 1
        s_h  = config["hidden_scalar"]    # 128
        v_h  = config["hidden_vector"]    # 16
        out  = config["output_dim"]       # 512
        L    = config["num_layers"]       # 3
        drop = config["dropout"]          # 0.1

        # Input projections to hidden dims
        self.node_in = GVP((s_n, v_n), (s_h, v_h))
        self.edge_in = GVP((s_e, v_e), (s_h // 4, v_e))

        s_e_h = s_h // 4

        # Stack of GVPConv layers
        self.layers = nn.ModuleList([
            GVPConv(
                node_in_dims=(s_h, v_h),
                edge_in_dims=(s_e_h, v_e),
                node_out_dims=(s_h, v_h),
                drop_rate=drop,
            )
            for _ in range(L)
        ])

        # Output: combine scalar features with vector norms → linear
        self.out_proj = nn.Sequential(
            nn.LayerNorm(s_h + v_h),
            nn.Linear(s_h + v_h, out),
            nn.ReLU(),
            nn.Linear(out, out),
        )

    def forward(self, batch):
        """
        batch : PyG Batch (or Data) object.
        Returns : (N, output_dim) tensor.
        """
        node_s = batch.x_scalar          # (N, 43)
        node_v = batch.x_vec             # (N, 3, 3)
        edge_index = batch.edge_index    # (2, E)
        edge_s = batch.edge_scalar       # (E, 16)
        edge_v = batch.edge_vec          # (E, 1, 3)

        # Project to hidden dims
        node_s, node_v = self.node_in(node_s, node_v)
        edge_s, edge_v = self.edge_in(edge_s, edge_v)

        # GVPConv message passing
        for layer in self.layers:
            node_s, node_v = layer(node_s, node_v, edge_index, edge_s, edge_v)

        # Build rotation-invariant output: scalars + vector norms
        v_norms = torch.norm(node_v, dim=-1)  # (N, v_h)
        out = torch.cat([node_s, v_norms], dim=-1)  # (N, s_h + v_h)
        return self.out_proj(out)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/models/test_encoder.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add prostrencoder/models/encoder.py tests/models/test_encoder.py
git commit -m "feat: ProStrEncoder GVP-GNN model producing per-residue embeddings"
```

---

### Task 9: Pre-Training Objective (Masked Residue Prediction)

**Files:**
- Create: `prostrencoder/training/objectives.py`
- Create: `tests/training/test_objectives.py`

- [ ] **Step 1: Write the failing test**

`tests/training/test_objectives.py`:

```python
import torch
import pytest
from prostrencoder.training.objectives import MaskedResidueLoss, apply_masking

def test_apply_masking_mask_rate():
    """~15% of residues should be masked."""
    N = 1000
    seq_idx = torch.randint(0, 20, (N,))
    masked_scalar, mask = apply_masking(seq_idx, torch.randn(N, 43), mask_rate=0.15)
    assert mask.dtype == torch.bool
    assert mask.shape == (N,)
    frac = mask.float().mean().item()
    assert 0.10 < frac < 0.20, f"Mask rate {frac:.3f} out of expected 10-20% range"

def test_apply_masking_zeros_masked_positions():
    """Masked positions in x_scalar should be zeroed out."""
    N = 50
    seq_idx = torch.randint(0, 20, (N,))
    x_scalar = torch.ones(N, 43)
    masked_scalar, mask = apply_masking(seq_idx, x_scalar, mask_rate=0.5)
    # Masked positions (mask=True) should have x_scalar = 0
    assert torch.all(masked_scalar[mask] == 0.0)
    # Unmasked positions should be unchanged
    assert torch.all(masked_scalar[~mask] == 1.0)

def test_apply_masking_returns_bool_tensor():
    seq_idx = torch.randint(0, 20, (20,))
    _, mask = apply_masking(seq_idx, torch.randn(20, 43))
    assert mask.dtype == torch.bool

def test_masked_residue_loss_shape():
    N = 30
    loss_fn = MaskedResidueLoss()
    logits = torch.randn(N, 21)
    targets = torch.randint(0, 21, (N,))
    mask = torch.zeros(N, dtype=torch.bool)
    mask[:5] = True
    loss = loss_fn(logits, targets, mask)
    assert loss.ndim == 0  # scalar

def test_masked_residue_loss_only_uses_masked():
    """Loss should only depend on masked positions."""
    N = 20
    loss_fn = MaskedResidueLoss()
    logits = torch.randn(N, 21)
    targets = torch.randint(0, 21, (N,))
    mask = torch.zeros(N, dtype=torch.bool)
    mask[:3] = True

    loss1 = loss_fn(logits, targets, mask)

    # Perturb unmasked logits — loss must not change
    logits2 = logits.clone()
    logits2[~mask] += 100.0
    loss2 = loss_fn(logits2, targets, mask)

    torch.testing.assert_close(loss1, loss2)

def test_masked_residue_loss_no_masked_returns_zero():
    N = 20
    loss_fn = MaskedResidueLoss()
    logits = torch.randn(N, 21)
    targets = torch.randint(0, 21, (N,))
    mask = torch.zeros(N, dtype=torch.bool)  # nothing masked
    loss = loss_fn(logits, targets, mask)
    assert loss.item() == 0.0
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/training/test_objectives.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `prostrencoder/training/objectives.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_masking(seq_idx: torch.Tensor, x_scalar: torch.Tensor,
                  mask_rate: float = 0.15):
    """
    BERT-style masking for residues.

    Randomly selects mask_rate fraction of residues.  Their scalar features
    are zeroed out (the model must predict their AA type from context).

    Args:
        seq_idx  : (N,) int64 — true AA indices (not modified)
        x_scalar : (N, D) float32 — will be copied and masked
        mask_rate: fraction of residues to mask

    Returns:
        masked_scalar : (N, D) — x_scalar with masked rows zeroed
        mask          : (N,) bool — True at masked positions
    """
    N = seq_idx.shape[0]
    n_mask = max(1, int(N * mask_rate))
    perm = torch.randperm(N, device=seq_idx.device)[:n_mask]

    mask = torch.zeros(N, dtype=torch.bool, device=seq_idx.device)
    mask[perm] = True

    masked_scalar = x_scalar.clone()
    masked_scalar[mask] = 0.0

    return masked_scalar, mask


class MaskedResidueLoss(nn.Module):
    """
    Cross-entropy loss over masked residue positions only.

    Forward args:
        logits  : (N, 21) — raw predictions for all residues
        targets : (N,) int64 — true AA indices (0–20)
        mask    : (N,) bool — True at positions to include in loss

    Returns scalar loss.  Returns 0.0 if no positions are masked.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if not mask.any():
            return logits.new_tensor(0.0)
        return F.cross_entropy(logits[mask], targets[mask])
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/training/test_objectives.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add prostrencoder/training/objectives.py tests/training/test_objectives.py
git commit -m "feat: masked residue prediction objective (BERT-style)"
```

---

### Task 10: Trainer and Training Head

**Files:**
- Create: `prostrencoder/models/gnn.py`  (prediction head)
- Create: `prostrencoder/training/trainer.py`
- Modify: `prostrencoder/models/encoder.py`  (expose hidden scalars for head)

- [ ] **Step 1: Write `prostrencoder/models/gnn.py` (residue prediction head)**

```python
import torch.nn as nn


class ResidueHead(nn.Module):
    """
    Linear head for masked residue type prediction.
    Takes per-residue embeddings and projects to 21-class logits.
    """

    def __init__(self, in_dim: int, num_classes: int = 21):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, embeddings):
        return self.linear(embeddings)
```

- [ ] **Step 2: Update `ProStrEncoder.forward` to optionally return pre-projection features**

Add a `return_hidden: bool = False` parameter so the trainer can share the scalar head without running the final projection twice.

In `prostrencoder/models/encoder.py`, replace the `forward` method:

```python
    def forward(self, batch, return_hidden: bool = False):
        node_s = batch.x_scalar
        node_v = batch.x_vec
        edge_index = batch.edge_index
        edge_s = batch.edge_scalar
        edge_v = batch.edge_vec

        node_s, node_v = self.node_in(node_s, node_v)
        edge_s, edge_v = self.edge_in(edge_s, edge_v)

        for layer in self.layers:
            node_s, node_v = layer(node_s, node_v, edge_index, edge_s, edge_v)

        v_norms = torch.norm(node_v, dim=-1)
        hidden = torch.cat([node_s, v_norms], dim=-1)  # (N, s_h + v_h)

        if return_hidden:
            return self.out_proj(hidden), hidden
        return self.out_proj(hidden)
```

- [ ] **Step 3: Verify existing encoder tests still pass after edit**

```bash
pytest tests/models/test_encoder.py -v
```

Expected: All PASS.

- [ ] **Step 4: Write `prostrencoder/training/trainer.py`**

```python
import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from prostrencoder.models.encoder import ProStrEncoder
from prostrencoder.models.gnn import ResidueHead
from prostrencoder.training.objectives import apply_masking, MaskedResidueLoss


class Trainer:
    """
    Self-supervised pre-training loop for ProStrEncoder via masked residue
    type prediction.

    Args:
        config       : dict loaded from configs/default.yaml
        train_dataset: ProteinDataset (preprocessed .pt files)
        val_dataset  : ProteinDataset or None
        device       : 'cuda' or 'cpu'
    """

    def __init__(self, config, train_dataset, val_dataset=None, device="cpu"):
        self.config = config
        self.device = torch.device(device)

        self.encoder = ProStrEncoder(config["model"]).to(self.device)

        hidden_dim = config["model"]["hidden_scalar"] + config["model"]["hidden_vector"]
        self.head = ResidueHead(hidden_dim).to(self.device)

        params = list(self.encoder.parameters()) + list(self.head.parameters())
        self.optimizer = AdamW(params,
                               lr=config["training"]["lr"],
                               weight_decay=config["training"]["weight_decay"])

        self.loss_fn = MaskedResidueLoss()
        self.mask_rate = config["training"]["mask_rate"]
        self.grad_clip = config["training"]["grad_clip"]

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
        ) if val_dataset is not None else None

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config["training"]["max_epochs"] * len(self.train_loader),
        )

        os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
        self.ckpt_dir = config["training"]["checkpoint_dir"]
        self.log_every = config["training"]["log_every"]

    def _forward_loss(self, batch):
        batch = batch.to(self.device)

        masked_scalar, mask = apply_masking(
            batch.seq_idx, batch.x_scalar, mask_rate=self.mask_rate
        )
        batch.x_scalar = masked_scalar

        _, hidden = self.encoder(batch, return_hidden=True)
        logits = self.head(hidden)           # (N_total, 21)
        loss = self.loss_fn(logits, batch.seq_idx, mask)
        return loss

    def train_epoch(self, epoch: int) -> float:
        self.encoder.train()
        self.head.train()
        total_loss = 0.0

        for step, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            loss = self._forward_loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.head.parameters()),
                self.grad_clip,
            )
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()

            if (step + 1) % self.log_every == 0:
                avg = total_loss / (step + 1)
                print(f"  Epoch {epoch} step {step+1}/{len(self.train_loader)}  loss={avg:.4f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def val_epoch(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.encoder.eval()
        self.head.eval()
        total = 0.0
        for batch in self.val_loader:
            total += self._forward_loss(batch).item()
        return total / len(self.val_loader)

    def save_checkpoint(self, epoch: int, val_loss: float):
        path = os.path.join(self.ckpt_dir, f"ckpt_epoch{epoch:03d}_val{val_loss:.4f}.pt")
        torch.save({
            "epoch": epoch,
            "encoder": self.encoder.state_dict(),
            "head": self.head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint: {path}")

    def fit(self):
        best_val = math.inf
        for epoch in range(1, self.config["training"]["max_epochs"] + 1):
            train_loss = self.train_epoch(epoch)
            val_loss   = self.val_epoch()
            print(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                self.save_checkpoint(epoch, val_loss)
```

- [ ] **Step 5: Write a smoke test for the training loop**

Add to `tests/training/test_objectives.py`:

```python
from torch_geometric.data import Data, Batch
from prostrencoder.models.encoder import ProStrEncoder
from prostrencoder.models.gnn import ResidueHead
from prostrencoder.training.objectives import apply_masking, MaskedResidueLoss

def _fake_batch(n=30, e=60):
    data = Data(
        seq_idx=torch.randint(0, 20, (n,)),
        x_scalar=torch.randn(n, 43),
        x_vec=torch.randn(n, 3, 3),
        edge_index=torch.randint(0, n, (2, e)),
        edge_scalar=torch.randn(e, 16),
        edge_vec=torch.randn(e, 1, 3),
    )
    return Batch.from_data_list([data])

def test_training_step_computes_gradient():
    config = {
        "num_layers": 1,
        "node_scalar_in": 43, "node_vector_in": 3,
        "edge_scalar_in": 16, "edge_vector_in": 1,
        "hidden_scalar": 32, "hidden_vector": 4,
        "output_dim": 64, "dropout": 0.0,
    }
    encoder = ProStrEncoder(config)
    head = ResidueHead(in_dim=32 + 4)
    loss_fn = MaskedResidueLoss()

    batch = _fake_batch()
    masked_scalar, mask = apply_masking(batch.seq_idx, batch.x_scalar)
    batch.x_scalar = masked_scalar

    _, hidden = encoder(batch, return_hidden=True)
    logits = head(hidden)
    loss = loss_fn(logits, batch.seq_idx, mask)
    loss.backward()

    for p in encoder.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()
```

- [ ] **Step 6: Run all training tests**

```bash
pytest tests/training/ -v
```

Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add prostrencoder/models/gnn.py prostrencoder/training/trainer.py \
        prostrencoder/models/encoder.py tests/training/test_objectives.py
git commit -m "feat: trainer with masked residue prediction and cosine LR schedule"
```

---

### Task 11: Training Script

**Files:**
- Create: `scripts/train.py`

- [ ] **Step 1: Write `scripts/train.py`**

```python
"""
Train ProStrEncoder with masked residue prediction.

Usage:
    python scripts/train.py --config configs/default.yaml \
                            --train_dir data/processed/train \
                            --val_dir   data/processed/val \
                            --device    cuda
"""
import argparse
import yaml
import torch

from prostrencoder.data.dataset import ProteinDataset
from prostrencoder.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/default.yaml")
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir",   default=None)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_ds = ProteinDataset(args.train_dir)
    val_ds   = ProteinDataset(args.val_dir) if args.val_dir else None

    print(f"Train: {len(train_ds)} proteins")
    if val_ds:
        print(f"Val:   {len(val_ds)} proteins")
    print(f"Device: {args.device}")

    trainer = Trainer(config, train_ds, val_ds, device=args.device)
    trainer.fit()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the script with synthetic data**

```bash
# Create two tiny preprocessed files for smoke testing
python - <<'EOF'
import torch
from torch_geometric.data import Data
import os

os.makedirs("data/processed/train", exist_ok=True)
os.makedirs("data/processed/val",   exist_ok=True)

for split, d in [("train", "data/processed/train"), ("val", "data/processed/val")]:
    for i in range(3):
        N, E = 30, 90
        data = Data(
            seq_idx=torch.randint(0, 20, (N,)),
            x_scalar=torch.randn(N, 43),
            x_vec=torch.randn(N, 3, 3),
            edge_index=torch.randint(0, N, (2, E)),
            edge_scalar=torch.randn(E, 16),
            edge_vec=torch.randn(E, 1, 3),
        )
        torch.save(data, f"{d}/prot_{i}.pt")
print("Synthetic data created.")
EOF

# Override max_epochs in a temp config
python -c "
import yaml
with open('configs/default.yaml') as f:
    c = yaml.safe_load(f)
c['training']['max_epochs'] = 2
c['training']['batch_size'] = 2
with open('/tmp/smoke.yaml', 'w') as f:
    yaml.dump(c, f)
"

python scripts/train.py --config /tmp/smoke.yaml \
                        --train_dir data/processed/train \
                        --val_dir data/processed/val \
                        --device cpu
```

Expected: Two epochs complete without error, checkpoint saved.

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "feat: training entry-point script for ProStrEncoder"
```

---

### Task 12: Downstream Evaluation

**Files:**
- Create: `scripts/eval_linear_probe.py`

This script freezes the encoder, trains a logistic regression head on frozen embeddings for SCOP fold classification (or any residue/protein-level label), and reports accuracy.

- [ ] **Step 1: Write `scripts/eval_linear_probe.py`**

```python
"""
Linear probe evaluation of frozen ProStrEncoder embeddings.

Expects preprocessed .pt files augmented with a `label` field (int64 scalar
per protein, e.g., SCOP fold class).

Usage:
    python scripts/eval_linear_probe.py \
        --checkpoint checkpoints/ckpt_epoch010_val0.1234.pt \
        --train_dir  data/scop/train \
        --test_dir   data/scop/test \
        --config     configs/default.yaml \
        --num_classes 1195 \
        --device     cuda
"""
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from prostrencoder.data.dataset import ProteinDataset
from prostrencoder.models.encoder import ProStrEncoder


def mean_pool(embeddings, batch_vec):
    """Mean-pool per-residue embeddings → one vector per protein."""
    N_prot = batch_vec.max().item() + 1
    out = embeddings.new_zeros(N_prot, embeddings.shape[-1])
    count = embeddings.new_zeros(N_prot)
    out.scatter_add_(0, batch_vec.unsqueeze(-1).expand_as(embeddings), embeddings)
    count.scatter_add_(0, batch_vec, embeddings.new_ones(embeddings.shape[0]))
    return out / (count.unsqueeze(-1) + 1e-8)


@torch.no_grad()
def extract_embeddings(encoder, loader, device):
    encoder.eval()
    all_emb, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        emb = encoder(batch)                          # (N_residues, D)
        pooled = mean_pool(emb, batch.batch)          # (B, D)
        all_emb.append(pooled.cpu())
        all_labels.append(batch.label.cpu())
    return torch.cat(all_emb), torch.cat(all_labels)


def train_linear_head(X_train, y_train, X_test, y_test,
                      num_classes, epochs=100, lr=1e-2):
    head = nn.Linear(X_train.shape[1], num_classes)
    opt  = AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        head.train()
        opt.zero_grad()
        loss = loss_fn(head(X_train), y_train)
        loss.backward()
        opt.step()

        if epoch % 20 == 0:
            head.eval()
            with torch.no_grad():
                preds = head(X_test).argmax(dim=-1)
                acc   = (preds == y_test).float().mean().item()
            print(f"  Epoch {epoch:3d}  test_acc={acc:.4f}")

    head.eval()
    with torch.no_grad():
        preds = head(X_test).argmax(dim=-1)
    return (preds == y_test).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--train_dir",   required=True)
    parser.add_argument("--test_dir",    required=True)
    parser.add_argument("--config",      default="configs/default.yaml")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    encoder = ProStrEncoder(config["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    for p in encoder.parameters():
        p.requires_grad_(False)

    train_ds = ProteinDataset(args.train_dir)
    test_ds  = ProteinDataset(args.test_dir)
    train_loader = DataLoader(train_ds, batch_size=32)
    test_loader  = DataLoader(test_ds,  batch_size=32)

    print("Extracting embeddings …")
    X_train, y_train = extract_embeddings(encoder, train_loader, device)
    X_test,  y_test  = extract_embeddings(encoder, test_loader,  device)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print("Training linear probe …")
    acc = train_linear_head(X_train, y_train, X_test, y_test,
                            num_classes=args.num_classes)
    print(f"\nFinal test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/eval_linear_probe.py
git commit -m "feat: linear probe evaluation script for downstream classification"
```

---

## Full Test Suite

Run all tests at once:

```bash
pytest tests/ -v --tb=short
```

Expected output (12 test files, all PASS):

```
tests/data/test_parser.py          ....  PASSED
tests/data/test_graph_builder.py   ......  PASSED
tests/data/test_features.py        .......  PASSED
tests/data/test_dataset.py         ....  PASSED
tests/models/test_rwse.py          ......  PASSED
tests/models/test_gvp.py           .......  PASSED
tests/models/test_encoder.py       ....  PASSED
tests/training/test_objectives.py  .......  PASSED
```

---

## Self-Review

### Spec coverage check

| Requirement | Task |
|---|---|
| Parse PDB/mmCIF → residue-level graph | Tasks 2, 3 |
| k-NN spatial graph (Cα–Cα) | Task 3 |
| Node features: AA identity, torsion angles | Task 4 |
| Node features: backbone frame (equivariant vectors) | Task 4 |
| Edge features: RBF distances, direction vectors | Task 4 |
| RWSE positional encoding | Task 5 |
| GNN backbone (GVP-style, SE(3)-equivariant) | Tasks 7–8 |
| Self-supervised pre-training: masked residue | Task 9 |
| Per-residue embeddings as output | Task 8 |
| Training pipeline with PDB dataset | Tasks 10–11 |
| Evaluation: downstream classification (linear probe) | Task 12 |

### Placeholder scan — none found

All code blocks are complete. No TBD / TODO / "implement later" patterns.

### Type consistency check

- `parse_structure()` returns `{"seq_idx": np.int64, "ca_coords": float32, "backbone_coords": float32}` — used consistently in Tasks 3, 4, 5, 6.
- `build_knn_graph()` returns `(edge_index: int64, edge_dist: float32)` — used consistently in Tasks 4, 6.
- `GVP(in_dims, out_dims)` signature used identically in Tasks 7 and 8 (`GVPConv`).
- `ProStrEncoder.forward(batch, return_hidden=False)` — introduced in Task 8, extended in Task 10 (adding `return_hidden`); Task 10 step 3 verifies backward compatibility.
- `ResidueHead(in_dim)` — `in_dim = hidden_scalar + hidden_vector` consistently used in Tasks 10 and test.
- `ProteinDataset` returns Data objects with fields `seq_idx, x_scalar, x_vec, edge_index, edge_scalar, edge_vec` — defined in Task 6, consumed in Tasks 9–12.

---

**Plan complete and saved to `docs/superpowers/plans/2026-06-01-prostrencoder.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
