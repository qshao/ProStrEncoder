# prostrencoder/training/trainer.py
import json
import math
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from prostrencoder.models.encoder import ProStrEncoder
from prostrencoder.models.gnn import ResidueHead
from prostrencoder.training.objectives import (
    apply_masking, MaskedResidueLoss, InverseFoldingLoss, sample_batch_mode,
)


class Trainer:
    """
    Two-phase pre-training loop for ProStrEncoder v2.

    Phase 1 (warmup_phase_fraction of total steps):
      - GVP + WalkEncoder + bridge + masked_head in optimizer (single group)
      - encoder.use_transformer = False -> bridge-only forward, no attention
      - Objective: masked residue prediction (Mode A) only

    Phase 2 (remaining steps):
      - encoder.use_transformer = True -> full transformer forward
      - Optimizer group 0: GVP + WalkEncoder at lr * gvp_lr_multiplier
      - Optimizer group 1: transformer layers + heads at lr
      - Alternating batches: Mode A (1 - inverse_fold_prob) or Mode B (inverse_fold_prob)

    Precision:
      - precision=fp32 -> no autocast (default, CPU-safe)
      - precision=bf16 -> torch.amp.autocast BF16 (CUDA only, no GradScaler needed)
    """

    def __init__(self, config: dict, train_dataset, val_dataset=None,
                 device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        train_cfg = config["training"]

        self.encoder = ProStrEncoder(config["model"]).to(self.device)

        hidden_dim = self.encoder.hidden_dim
        self.masked_head   = ResidueHead(hidden_dim).to(self.device)
        self.inv_fold_head = ResidueHead(hidden_dim).to(self.device)

        self.masked_loss_fn   = MaskedResidueLoss()
        self.inv_fold_loss_fn = InverseFoldingLoss()

        self.mask_rate        = train_cfg["mask_rate"]
        self.grad_clip        = train_cfg["grad_clip"]
        self.inv_fold_prob    = train_cfg.get("inverse_fold_prob", 0.3)
        self.inv_fold_weight  = train_cfg.get("inverse_fold_weight", 1.0)
        self.precision        = train_cfg.get("precision", "fp32")
        self.log_every        = train_cfg["log_every"]
        self.save_every_steps = train_cfg.get("save_every_steps", 1000)

        num_workers = train_cfg.get("num_workers", 4)
        pin_memory  = (self.device.type == "cuda")

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg["batch_size"],
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        ) if val_dataset is not None else None

        total_steps = train_cfg["max_epochs"] * len(self.train_loader)
        warmup_steps = max(1, int(total_steps * train_cfg.get("warmup_phase_fraction", 0.05)))
        self._warmup_steps   = warmup_steps
        self._total_steps    = total_steps
        self._gvp_lr_mult    = train_cfg.get("gvp_lr_multiplier", 0.1)
        self._peak_lr        = train_cfg["lr"]
        self._weight_decay   = train_cfg["weight_decay"]
        self._step           = 0
        self._phase          = 1

        # Phase 1 optimizer: GVP + WalkEncoder + bridge + masked_head + out_proj
        self._build_phase1_optimizer()

        # Cosine LR schedule runs across both phases without reset
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)
        self.ckpt_dir = train_cfg["checkpoint_dir"]

    # ── Optimizer builders ─────────────────────────────────────────────────────

    def _raw_encoder(self):
        """Return the underlying ProStrEncoder, unwrapping DDP/FSDP if present."""
        enc = self.encoder
        return enc.module if hasattr(enc, "module") else enc

    def _phase1_params(self):
        """All params active during Phase 1."""
        enc = self._raw_encoder()
        params = (list(enc.walk_sampler.parameters()) +
                  list(enc.walk_encoder.parameters()) +
                  list(enc.node_in.parameters()) +
                  list(enc.edge_in.parameters()) +
                  list(enc.layers.parameters()))
        if enc.transformer is not None:
            params += list(enc.transformer.bridge.parameters())
        params += (list(enc.out_proj.parameters()) +
                   list(self.masked_head.parameters()))
        return params

    def _build_phase1_optimizer(self):
        params = self._phase1_params()
        self.optimizer = AdamW(params, lr=self._peak_lr,
                               weight_decay=self._weight_decay)

    def _transition_to_phase2(self):
        """Switch to Phase 2: activate transformer, add second LR group."""
        enc = self._raw_encoder()
        if enc.transformer is None:
            # No transformer configured — stay as single group
            return
        enc.use_transformer = True   # accesses buffer via property setter
        self._phase = 2

        # Group 0: GVP + WalkEncoder + bridge (conservative LR)
        gvp_params = (list(enc.walk_sampler.parameters()) +
                      list(enc.walk_encoder.parameters()) +
                      list(enc.node_in.parameters()) +
                      list(enc.edge_in.parameters()) +
                      list(enc.layers.parameters()) +
                      list(enc.transformer.bridge.parameters()))

        # Group 1: transformer layers + norm + both heads + out_proj (full LR)
        tx_params = (list(enc.transformer.layers.parameters()) +
                     list(enc.transformer.norm.parameters()) +
                     list(enc.out_proj.parameters()) +
                     list(self.masked_head.parameters()) +
                     list(self.inv_fold_head.parameters()))

        current_lr = self.scheduler.get_last_lr()[0] if self._step > 0 else self._peak_lr

        # Build new optimizer with param groups for phase 2
        old_state = {id(p): s for p, s in self.optimizer.state.items()}
        new_opt = AdamW([
            {"params": gvp_params, "lr": current_lr * self._gvp_lr_mult},
            {"params": tx_params,  "lr": current_lr},
        ], weight_decay=self._weight_decay)

        # Transfer accumulated Adam momentum/second-moment state for surviving GVP params
        for group in new_opt.param_groups:
            for p in group["params"]:
                if id(p) in old_state:
                    new_opt.state[p] = old_state[id(p)]

        self.optimizer = new_opt

        # Rebuild scheduler for the remaining step budget so cosine decay continues
        remaining = max(1, self._total_steps - self._step)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=remaining)

    # ── Forward + loss ──────────────────────────────────────────────────────────

    @property
    def _use_amp(self) -> bool:
        return self.precision == "bf16" and self.device.type == "cuda"

    def _forward_loss(self, batch):
        batch = batch.to(self.device)
        targets = batch.seq_idx.clone()

        # In Phase 1 always use Mode A; in Phase 2 alternate
        if self._phase == 1:
            mode = "A"
        else:
            mode = sample_batch_mode(self.inv_fold_prob)

        if mode == "A":
            masked_scalar, masked_seq_idx, mask = apply_masking(
                batch.seq_idx, batch.x_scalar, mask_rate=self.mask_rate
            )
            batch.x_scalar = masked_scalar
            batch.seq_idx  = masked_seq_idx

        amp_dtype = torch.bfloat16 if self._use_amp else torch.float32
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self._use_amp):
            _, hidden = self.encoder(batch, return_hidden=True)
            if mode == "A":
                logits = self.masked_head(hidden)
                loss   = self.masked_loss_fn(logits, targets, mask)
            else:
                logits  = self.inv_fold_head(hidden)
                weights = getattr(batch, "plddt", None)
                loss    = self.inv_fold_loss_fn(logits, targets, weights=weights)
                loss    = loss * self.inv_fold_weight

        return loss, mode

    # ── Training loop ──────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> float:
        self.encoder.train()
        self.masked_head.train()
        self.inv_fold_head.train()
        total_loss = 0.0

        for step_in_epoch, batch in enumerate(self.train_loader):
            # Phase transition check
            if self._phase == 1 and self._step >= self._warmup_steps:
                self._transition_to_phase2()

            self.optimizer.zero_grad()
            loss, mode = self._forward_loss(batch)
            loss.backward()

            all_params = []
            for pg in self.optimizer.param_groups:
                all_params += pg["params"]
            nn.utils.clip_grad_norm_(all_params, self.grad_clip)

            self.optimizer.step()
            self.scheduler.step()
            self._step += 1
            total_loss += loss.item()

            if (step_in_epoch + 1) % self.log_every == 0:
                lrs = [pg["lr"] for pg in self.optimizer.param_groups]
                record = {
                    "step": self._step, "epoch": epoch, "mode": mode,
                    "loss": round(loss.item(), 6),
                    "perplexity": round(math.exp(min(loss.item(), 20)), 4),
                    "lr": lrs[-1],
                    "phase": self._phase,
                }
                print(json.dumps(record))

            if self._step % self.save_every_steps == 0:
                self.save_checkpoint(epoch, float("nan"))

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def val_epoch(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.encoder.eval()
        self.masked_head.eval()
        self.inv_fold_head.eval()
        total = 0.0
        for batch in self.val_loader:
            loss, _ = self._forward_loss(batch)
            total += loss.item()
        return total / len(self.val_loader)

    def save_checkpoint(self, epoch: int, val_loss: float):
        tag = (f"step{self._step:07d}" if math.isnan(val_loss)
               else f"epoch{epoch:03d}_val{val_loss:.4f}")
        path = os.path.join(self.ckpt_dir, f"ckpt_{tag}.pt")
        torch.save({
            "epoch": epoch,
            "step": self._step,
            "phase": self._phase,
            "encoder": self.encoder.state_dict(),
            "masked_head": self.masked_head.state_dict(),
            "inv_fold_head": self.inv_fold_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
        print(json.dumps({"event": "checkpoint", "path": path}))

    def fit(self):
        best_val = math.inf
        for epoch in range(1, self.config["training"]["max_epochs"] + 1):
            train_loss = self.train_epoch(epoch)
            val_loss   = self.val_epoch()
            print(json.dumps({"event": "epoch", "epoch": epoch,
                               "train_loss": round(train_loss, 6),
                               "val_loss": round(val_loss, 6) if not math.isnan(val_loss) else None,
                               "phase": self._phase}))
            if val_loss < best_val:
                best_val = val_loss
                self.save_checkpoint(epoch, val_loss)
