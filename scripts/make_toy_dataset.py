"""
Generate a toy protein dataset for testing the training pipeline.

Creates synthetic protein structures — a configurable mix of ideal alpha-helix
and random-coil geometry — and processes them through the full preprocessing
pipeline (graph construction, feature extraction) to produce .pt files
compatible with ProteinDataset and the Trainer.

Generation parameters are read from config.toy_data; CLI args override them.

Usage (YAML defaults):
    python scripts/make_toy_dataset.py --config configs/default.yaml

Usage (override specific values):
    python scripts/make_toy_dataset.py --config configs/default.yaml \\
        --n_train 200 --n_val 40 --max_len 200 --seed 99

Then train:
    python scripts/train.py --config configs/default.yaml --device cpu
"""
import argparse
import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from torch_geometric.data import Data
from prostrencoder.data.features import build_pyg_data
from prostrencoder.data.graph_builder import build_knn_graph


# ── Structure generators ───────────────────────────────────────────────────────

def _helix_ca(n: int, rise: float = 1.5, radius: float = 2.3,
              turn_deg: float = 100.0) -> np.ndarray:
    """Ideal alpha-helix Cα coordinates. Shape (n, 3)."""
    t = np.arange(n, dtype=np.float32)
    ang = np.deg2rad(turn_deg) * t
    return np.stack([radius * np.cos(ang),
                     radius * np.sin(ang),
                     rise * t], axis=1)


def _coil_ca(n: int, bond: float = 3.8, rng: np.random.Generator = None) -> np.ndarray:
    """Random-coil Cα walk with fixed bond length. Shape (n, 3)."""
    if rng is None:
        rng = np.random.default_rng()
    coords = np.zeros((n, 3), dtype=np.float32)
    for i in range(1, n):
        d = rng.standard_normal(3).astype(np.float32)
        coords[i] = coords[i - 1] + bond * d / (np.linalg.norm(d) + 1e-8)
    return coords


def _backbone_from_ca(ca: np.ndarray) -> np.ndarray:
    """
    Approximate N, CA, C, O positions from Cα coords.
    Returns (N, 4, 3) float32. Column order: N=0, CA=1, C=2, O=3.
    """
    n = len(ca)
    bb = np.zeros((n, 4, 3), dtype=np.float32)
    bb[:, 1] = ca

    for i in range(n):
        if i < n - 1:
            fwd = ca[i + 1] - ca[i]
        else:
            fwd = ca[i] - ca[i - 1]
        fwd = fwd / (np.linalg.norm(fwd) + 1e-8)

        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(np.dot(fwd, ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        perp = np.cross(fwd, ref).astype(np.float32)
        perp /= np.linalg.norm(perp) + 1e-8

        bb[i, 0] = ca[i] - 1.46 * fwd + 0.30 * perp   # N
        bb[i, 2] = ca[i] + 1.52 * fwd - 0.30 * perp   # C
        bb[i, 3] = bb[i, 2] + 1.23 * perp              # O

    return bb


# ── Single-protein builder ─────────────────────────────────────────────────────

def make_protein(n_residues: int, helix_fraction: float = 0.5,
                 k: int = 30, rng: np.random.Generator = None) -> Data:
    """
    Synthesise one protein as a PyG Data object ready for ProteinDataset.

    All features match the format produced by scripts/preprocess.py:
        seq_idx    (N,)    int64  — random AA indices 0–19
        x_scalar   (N,27)  float  — AA one-hot (21) + torsion (6)
        x_vec      (N,3,3) float  — backbone frame
        edge_index (2,E)   int64
        edge_scalar (E,16) float  — RBF distances
        edge_vec   (E,1,3) float  — direction vectors
    """
    if rng is None:
        rng = np.random.default_rng()

    seq_idx = rng.integers(0, 20, size=n_residues).astype(np.int64)

    n_helix = max(0, int(n_residues * helix_fraction))
    n_coil  = n_residues - n_helix

    segments = []
    if n_helix > 0:
        segments.append(_helix_ca(n_helix))
    if n_coil > 0:
        coil = _coil_ca(n_coil, rng=rng)
        if segments:
            coil = coil - coil[0] + segments[-1][-1] + np.array([3.8, 0., 0.], np.float32)
        segments.append(coil)

    ca = np.concatenate(segments, axis=0).astype(np.float32)
    bb = _backbone_from_ca(ca)

    edge_index, edge_dist = build_knn_graph(ca, k=k)

    return build_pyg_data(seq_idx, ca, bb, edge_index, edge_dist)


# ── Split generator ────────────────────────────────────────────────────────────

def generate_split(name: str, n: int, out_dir: str,
                   min_len: int, max_len: int, k: int,
                   seed: int) -> None:
    split_dir = os.path.join(out_dir, name)
    os.makedirs(split_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    for i in tqdm(range(n), desc=f"{name:5s}", unit="prot"):
        n_res      = int(rng.integers(min_len, max_len + 1))
        helix_frac = float(rng.uniform(0.0, 1.0))
        data = make_protein(n_res, helix_fraction=helix_frac, k=k, rng=rng)
        torch.save(data, os.path.join(split_dir, f"protein_{i:04d}.pt"))

    print(f"  {name:5s} — {n} proteins saved to {split_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a toy protein dataset for pipeline testing."
    )
    parser.add_argument("--config",   default="configs/default.yaml",
                        help="YAML config file (provides toy_data defaults)")
    # All generation params are optional; YAML provides defaults
    parser.add_argument("--out_dir",  default=None,
                        help="Root output directory (overrides config.toy_data.out_dir)")
    parser.add_argument("--n_train",  type=int, default=None,
                        help="Number of training proteins")
    parser.add_argument("--n_val",    type=int, default=None)
    parser.add_argument("--n_test",   type=int, default=None)
    parser.add_argument("--min_len",  type=int, default=None,
                        help="Minimum residues per protein")
    parser.add_argument("--max_len",  type=int, default=None,
                        help="Maximum residues per protein")
    parser.add_argument("--k",        type=int, default=None,
                        help="k-NN neighbours for graph construction")
    parser.add_argument("--seed",     type=int, default=None,
                        help="Base random seed (val/test use seed+1, seed+2)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides YAML; YAML overrides built-in fallbacks
    toy = config.get("toy_data", {})
    data_cfg = config.get("data", {})

    def _get(attr, fallback):
        cli_val = getattr(args, attr)
        return cli_val if cli_val is not None else toy.get(attr, fallback)

    out_dir = args.out_dir or toy.get("out_dir", "data/toy")
    n_train = _get("n_train", 100)
    n_val   = _get("n_val",   20)
    n_test  = _get("n_test",  20)
    min_len = _get("min_len", 40)
    max_len = _get("max_len", 150)
    k       = args.k if args.k is not None else data_cfg.get("k_neighbors", toy.get("k", 30))
    seed    = _get("seed", 42)

    print(f"\nToy dataset  →  {out_dir}")
    print(f"  Split sizes : {n_train} train / {n_val} val / {n_test} test")
    print(f"  Length range: {min_len}–{max_len} residues")
    print(f"  Graph k-NN  : k={k}   seed={seed}\n")

    generate_split("train", n_train, out_dir, min_len, max_len, k, seed)
    generate_split("val",   n_val,   out_dir, min_len, max_len, k, seed + 1)
    generate_split("test",  n_test,  out_dir, min_len, max_len, k, seed + 2)

    print(f"\nDone. Run training with:")
    print(f"  python scripts/train.py --config {args.config}")


if __name__ == "__main__":
    main()
