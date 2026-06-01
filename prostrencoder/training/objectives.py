import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_masking(seq_idx: torch.Tensor, x_scalar: torch.Tensor,
                  mask_rate: float = 0.15):
    """
    BERT-style masking for residues.

    Randomly selects mask_rate fraction of residues.  Their scalar features
    are zeroed out (the model must predict their AA type from context).

    Args:
        seq_idx  : (N,) int64 — true AA indices (not modified)
        x_scalar : (N, D) float32 — will be copied and masked
        mask_rate: fraction of residues to mask

    Returns:
        masked_scalar : (N, D) — x_scalar with masked rows zeroed
        mask          : (N,) bool — True at masked positions
    """
    N = seq_idx.shape[0]
    n_mask = max(1, int(N * mask_rate))
    perm = torch.randperm(N, device=seq_idx.device)[:n_mask]

    mask = torch.zeros(N, dtype=torch.bool, device=seq_idx.device)
    mask[perm] = True

    masked_scalar = x_scalar.clone()
    masked_scalar[mask] = 0.0

    return masked_scalar, mask


class MaskedResidueLoss(nn.Module):
    """
    Cross-entropy loss over masked residue positions only.

    Forward args:
        logits  : (N, 21) — raw predictions for all residues
        targets : (N,) int64 — true AA indices (0–20)
        mask    : (N,) bool — True at positions to include in loss

    Returns scalar loss.  Returns 0.0 if no positions are masked.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if not mask.any():
            return logits.new_tensor(0.0)
        return F.cross_entropy(logits[mask], targets[mask])
