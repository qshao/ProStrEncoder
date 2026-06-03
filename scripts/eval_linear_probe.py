"""
Linear probe evaluation of frozen ProStrEncoder embeddings.

Expects preprocessed .pt files augmented with a `label` field (int64 scalar
per protein, e.g., SCOP fold class).

Probe hyperparameters (epochs, lr, weight_decay) are read from the YAML
config under the `evaluation:` section; CLI args override them.

Usage:
    python scripts/eval_linear_probe.py \\
        --checkpoint checkpoints/ckpt_epoch010_val0.1234.pt \\
        --train_dir  data/scop/train \\
        --test_dir   data/scop/test \\
        --config     configs/default.yaml \\
        --num_classes 1195 \\
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
                      num_classes, epochs, lr, weight_decay):
    """Train a linear probe on frozen embeddings and return test accuracy."""
    head    = nn.Linear(X_train.shape[1], num_classes)
    opt     = AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
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
    parser = argparse.ArgumentParser(
        description="Linear probe evaluation of frozen ProStrEncoder embeddings."
    )
    parser.add_argument("--checkpoint",    required=True,
                        help="Path to encoder checkpoint .pt file")
    parser.add_argument("--train_dir",     required=True,
                        help="Labelled train split directory")
    parser.add_argument("--test_dir",      required=True,
                        help="Labelled test split directory")
    parser.add_argument("--config",        default="configs/default.yaml",
                        help="YAML config (model arch + evaluation defaults)")
    parser.add_argument("--num_classes",   type=int, required=True,
                        help="Number of classification labels")
    parser.add_argument("--device",        default="cpu")
    # Optional CLI overrides for probe hyperparams
    parser.add_argument("--probe_epochs",       type=int,   default=None,
                        help="Probe training epochs (overrides config.evaluation.probe_epochs)")
    parser.add_argument("--probe_lr",           type=float, default=None,
                        help="Probe learning rate (overrides config.evaluation.probe_lr)")
    parser.add_argument("--probe_weight_decay", type=float, default=None,
                        help="Probe weight decay (overrides config.evaluation.probe_weight_decay)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Probe hyperparams: CLI > YAML > built-in fallback
    eval_cfg = config.get("evaluation", {})
    probe_epochs  = args.probe_epochs       if args.probe_epochs       is not None else eval_cfg.get("probe_epochs",       100)
    probe_lr      = args.probe_lr           if args.probe_lr           is not None else eval_cfg.get("probe_lr",           0.01)
    probe_wd      = args.probe_weight_decay if args.probe_weight_decay is not None else eval_cfg.get("probe_weight_decay", 1e-4)

    print(f"Probe config: epochs={probe_epochs}  lr={probe_lr}  weight_decay={probe_wd}")

    device = torch.device(args.device)
    encoder = ProStrEncoder(config["model"]).to(device)
    # weights_only=False required: v2 checkpoints contain a config dict
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    # Restore transformer activation from checkpoint phase (v2 only)
    if ckpt.get("phase", 1) == 2:
        encoder.use_transformer = True
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
    acc = train_linear_head(
        X_train, y_train, X_test, y_test,
        num_classes=args.num_classes,
        epochs=probe_epochs,
        lr=probe_lr,
        weight_decay=probe_wd,
    )
    print(f"\nFinal test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
