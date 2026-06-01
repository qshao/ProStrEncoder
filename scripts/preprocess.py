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
