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
