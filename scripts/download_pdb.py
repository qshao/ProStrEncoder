# scripts/download_pdb.py
"""
Download PDB structures from RCSB and apply per-chain quality filters.

Filters (all configurable via YAML):
  - Resolution <= resolution_cutoff Angstroms (X-ray; cryo-EM always accepted)
  - >= min_chain_coverage fraction of residues have all 4 backbone atoms
  - Chain length within [min_len, max_len]

After filtering, runs MMseqs2 at cluster_seqid identity to split train/val
and prevent homolog leakage.

Usage:
    python scripts/download_pdb.py --config configs/medium.yaml
    python scripts/download_pdb.py --config configs/medium.yaml --skip_download
"""
import argparse
import os
import random
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
    print(f"Syncing PDB from RCSB -> {pdb_dir}")
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

    # Resolution check via BioPython header
    try:
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("X", pdb_path)
        resolution = structure.header.get("resolution")
        if resolution is not None and float(resolution) > resolution_cutoff:
            return False
    except Exception:
        pass   # If we cannot read resolution, do not filter

    return True


def _write_fasta(chain_paths: list, fasta_path: str):
    """Write a FASTA file from chain paths for MMseqs2 clustering."""
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
    Cluster sequences with MMseqs2. Returns dict mapping repr -> [members].
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
            parts = line.strip().split("\t")
            if len(parts) == 2:
                repr_id, member = parts
                clusters.setdefault(repr_id, []).append(member)
    return clusters


def _split_clusters(clusters: dict, val_frac: float = 0.05,
                    seed: int = 42) -> tuple:
    """Split cluster representatives into train and val sets."""
    reprs = sorted(clusters.keys())
    rng = random.Random(seed)
    rng.shuffle(reprs)
    n_val = max(1, int(len(reprs) * val_frac))
    val_reprs   = set(reprs[:n_val])

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
    parser.add_argument("--seed",          type=int,   default=42,
                        help="Random seed for train/val cluster split")
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
        for fname in files:
            if fname.endswith((".cif", ".cif.gz", ".pdb", ".pdb.gz")):
                pdb_files.append(os.path.join(root, fname))
    print(f"Found {len(pdb_files)} structure files")

    # Apply quality filters
    print("Applying quality filters ...")
    passing = [p for p in pdb_files
               if _chain_passes_filters(p, res_cut, min_cov, min_len, max_len)]
    print(f"Passing filters: {len(passing)}/{len(pdb_files)}")

    # Write FASTA and cluster
    mmseqs_dir = os.path.join(out_base, "mmseqs")
    os.makedirs(mmseqs_dir, exist_ok=True)
    fasta_path = os.path.join(mmseqs_dir, "chains.fasta")
    _write_fasta(passing, fasta_path)
    print(f"FASTA written: {fasta_path} ({len(passing)} sequences)")

    print(f"Running MMseqs2 at {seqid * 100:.0f}% identity ...")
    clusters = _run_mmseqs2(fasta_path, seqid, mmseqs_dir)
    print(f"Clusters: {len(clusters)}")

    train_paths, val_paths = _split_clusters(
        clusters, val_frac=args.val_frac, seed=args.seed
    )
    print(f"Train chains: {len(train_paths)}, Val chains: {len(val_paths)}")

    # Write split manifest files
    os.makedirs(out_base, exist_ok=True)
    for split, paths in [("train", train_paths), ("val", val_paths)]:
        manifest = os.path.join(out_base, f"{split}_manifest.txt")
        with open(manifest, "w") as mf:
            mf.write("\n".join(paths))
        print(f"Written: {manifest}")

    print("\nNext step: run scripts/preprocess.py for each split")
    print(f"  python scripts/preprocess.py --config {args.config} "
          f"--pdb_dir {pdb_dir} --out_dir {out_base}/train")


if __name__ == "__main__":
    main()
