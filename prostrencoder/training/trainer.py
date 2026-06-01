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
