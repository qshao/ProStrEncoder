# scripts/eval_inverse_folding.py
"""
Evaluate the inverse folding head's native sequence recovery rate.

For each held-out structure: given the full 3D backbone, the inverse folding
head predicts a sequence. Native sequence recovery = fraction of positions
where the predicted AA matches the true AA.

Usage:
    python scripts/eval_inverse_folding.py \\
        --checkpoint checkpoints/medium/ckpt_epoch010_val1.2345.pt \\
        --test_dir   data/processed/val \\
        --config     configs/medium.yaml \\
        --device     cuda
"""
import argparse
import json
import math

import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

from prostrencoder.data.dataset import ProteinDataset
from prostrencoder.models.encoder import ProStrEncoder
from prostrencoder.models.gnn import ResidueHead


@torch.no_grad()
def evaluate(encoder: ProStrEncoder, inv_fold_head: ResidueHead,
             loader: DataLoader, device: torch.device) -> dict:
    encoder.eval()
    inv_fold_head.eval()

    total_residues   = 0
    correct_residues = 0
    total_ce_loss    = 0.0
    n_batches        = 0

    for batch in loader:
        batch   = batch.to(device)
        targets = batch.seq_idx.clone()

        # Inverse folding: full structure in, no masking
        _, hidden = encoder(batch, return_hidden=True)
        logits    = inv_fold_head(hidden)             # (N, 21)

        # Per-residue accuracy
        preds   = logits.argmax(dim=-1)               # (N,)
        correct = (preds == targets).sum().item()
        correct_residues += correct
        total_residues   += targets.shape[0]

        # Per-residue cross-entropy (negative log-likelihood)
        ce = F.cross_entropy(logits, targets, reduction="mean")
        total_ce_loss += ce.item()
        n_batches += 1

    recovery   = correct_residues / max(total_residues, 1)
    mean_ce    = total_ce_loss / max(n_batches, 1)
    perplexity = math.exp(min(mean_ce, 20))

    return {
        "sequence_recovery": round(recovery, 4),
        "mean_ce_loss":      round(mean_ce, 6),
        "perplexity":        round(perplexity, 4),
        "total_residues":    total_residues,
        "correct_residues":  correct_residues,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate inverse folding sequence recovery."
    )
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--test_dir",    required=True,
                        help="Directory of processed .pt structure files")
    parser.add_argument("--config",      default="configs/medium.yaml")
    parser.add_argument("--batch_size",  type=int, default=None,
                        help="Override config batch_size")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device     = torch.device(args.device)
    model_cfg  = config["model"]
    batch_size = args.batch_size or config["training"]["batch_size"]

    # Build model
    encoder = ProStrEncoder(model_cfg).to(device)
    encoder.use_transformer = True   # Phase 2 full forward

    hidden_dim    = encoder.hidden_dim
    inv_fold_head = ResidueHead(hidden_dim).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    inv_fold_head.load_state_dict(ckpt["inv_fold_head"])
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  trained for {ckpt.get('step', '?')} steps, phase {ckpt.get('phase', '?')}")

    dataset = ProteinDataset(args.test_dir)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0)
    print(f"Evaluating on {len(dataset)} structures from {args.test_dir}")

    results = evaluate(encoder, inv_fold_head, loader, device)
    print(json.dumps(results, indent=2))

    recovery_pct = results["sequence_recovery"] * 100
    target_pct   = 35.0
    if recovery_pct >= target_pct:
        print(f"\nRecovery {recovery_pct:.1f}% meets target (>={target_pct:.0f}%)")
    else:
        print(f"\nRecovery {recovery_pct:.1f}% below target ({target_pct:.0f}%)")


if __name__ == "__main__":
    main()
