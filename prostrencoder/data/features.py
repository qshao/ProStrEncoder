import numpy as np
import torch
from torch_geometric.data import Data


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
    # Guard: NaN inputs (missing backbone atoms) → return 0 instead of propagating NaN.
    # The norm-based guard below uses `< 1e-6` which evaluates False for NaN.
    if (np.any(np.isnan(p0)) or np.any(np.isnan(p1)) or
            np.any(np.isnan(p2)) or np.any(np.isnan(p3))):
        return 0.0
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

    # Belt-and-suspenders: clamp any residual NaN to 0 (sin/cos of 0 = 0/1)
    return np.nan_to_num(out, nan=0.0)


def rbf_encoding(distances: np.ndarray, num_rbf: int = 16,
                 d_min: float = 0.0, d_max: float = 20.0) -> np.ndarray:
    """
    Gaussian RBF encoding of edge distances.

    distances : (E,) float32
    Returns   : (E, num_rbf) float32
    """
    centers = np.linspace(d_min, d_max, num_rbf, dtype=np.float32)
    sigma = (d_max - d_min) / max(num_rbf - 1, 1)
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


def build_pyg_data(
    seq_idx: np.ndarray,
    ca_coords: np.ndarray,
    backbone_coords: np.ndarray,
    edge_index: np.ndarray,
    edge_dist: np.ndarray,
) -> Data:
    """
    Assemble a PyG Data object from raw protein arrays.

    Single source of truth for feature computation used by both
    scripts/preprocess.py (real PDB data) and scripts/make_toy_dataset.py
    (synthetic data), ensuring identical feature layouts.

    Args:
        seq_idx          : (N,) int64  — AA indices 0–20
        ca_coords        : (N, 3) float32
        backbone_coords  : (N, 4, 3) float32  — N, CA, C, O per residue
        edge_index       : (2, E) int64
        edge_dist        : (E,) float32

    Returns PyG Data with fields:
        seq_idx, x_scalar (N,27), x_vec (N,3,3),
        edge_index, edge_scalar (E,16), edge_vec (E,1,3)
    """
    onehot  = aa_one_hot(seq_idx)                                   # (N, 21)
    torsion = compute_torsion_angles(backbone_coords)               # (N, 6)
    frame   = compute_backbone_frame(backbone_coords)               # (N, 3, 3)
    rbf     = rbf_encoding(edge_dist)                               # (E, 16)
    dirs    = compute_edge_directions(ca_coords, edge_index,
                                      edge_dist)                    # (E, 3)

    x_scalar = np.concatenate([onehot, torsion], axis=1).astype(np.float32)
    e_vec    = dirs[:, np.newaxis, :].astype(np.float32)            # (E, 1, 3)

    return Data(
        seq_idx=torch.from_numpy(seq_idx),
        x_scalar=torch.from_numpy(x_scalar),
        x_vec=torch.from_numpy(frame),
        edge_index=torch.from_numpy(edge_index),
        edge_scalar=torch.from_numpy(rbf),
        edge_vec=torch.from_numpy(e_vec),
    )
