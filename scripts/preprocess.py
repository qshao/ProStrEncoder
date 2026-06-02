"""
Convert a directory of PDB files to preprocessed PyG Data objects.

x_scalar stores only AA one-hot (21) + torsion angles (6) = 27 dims.
The 64-dim walk embedding is computed online inside ProStrEncoder.forward()
and is no longer stored in the .pt file.

YAML config provides default paths and k-NN value; CLI args override them.

Usage (YAML defaults):
    python scripts/preprocess.py --config configs/default.yaml

Usage (full CLI override):
    python scripts/preprocess.py --config configs/default.yaml \\
        --pdb_dir /data/rcsb --out_dir data/processed --k 20
"""
import argparse
import os

import torch
import yaml
from torch_geometric.data import Data
from tqdm import tqdm

from prostrencoder.data.features import build_pyg_data
from prostrencoder.data.graph_builder import build_knn_graph
from prostrencoder.data.parser import parse_structure


def preprocess_one(pdb_path: str, k: int) -> Data:
    parsed = parse_structure(pdb_path)
    if len(parsed["seq_idx"]) < 4:
        raise ValueError(f"Too few residues in {pdb_path}")

    edge_index, edge_dist = build_knn_graph(parsed["ca_coords"], k=k)

    return build_pyg_data(
        parsed["seq_idx"], parsed["ca_coords"], parsed["backbone_coords"],
        edge_index, edge_dist,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess PDB/mmCIF files into PyG .pt graph objects."
    )
    parser.add_argument("--config",  default="configs/default.yaml",
                        help="YAML config file (provides default paths and k)")
    parser.add_argument("--pdb_dir", default=None,
                        help="Input directory of .pdb/.cif files "
                             "(overrides config.data.pdb_dir)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for .pt files "
                             "(overrides config.data.processed_dir)")
    parser.add_argument("--k",       type=int, default=None,
                        help="k-NN neighbours (overrides config.data.k_neighbors)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg  = config.get("data", {})
    pdb_dir   = args.pdb_dir or data_cfg.get("pdb_dir", "data/pdb")
    out_dir   = args.out_dir or data_cfg.get("processed_dir", "data/processed")
    k         = args.k if args.k is not None else data_cfg.get("k_neighbors", 30)
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


if __name__ == "__main__":
    main()
