import torch
import torch.nn as nn
import torch.nn.functional as F

# Token 22: dedicated [MASK] token fed to WalkSampler for masked residues.
# Vocab layout: 0-20 = AA types, 21 = CLS (WalkSampler), 22 = MASK.
MASK_AA: int = 22


def apply_masking(seq_idx: torch.Tensor, x_scalar: torch.Tensor,
                  mask_rate: float = 0.15):
    """
    BERT-style masking for residues.

    Randomly selects mask_rate fraction of residues. Their scalar features are
    zeroed and their seq_idx entries are replaced with MASK_AA (token 22) so
    that WalkSampler cannot leak the true AA identity through walk sequences.

    Args:
        seq_idx  : (N,) int64 — true AA indices 0–20
        x_scalar : (N, D) float32 — will be copied and masked
        mask_rate: fraction of residues to mask

    Returns:
        masked_scalar  : (N, D) — x_scalar with masked rows zeroed
        masked_seq_idx : (N,) int64 — seq_idx with masked positions = MASK_AA
        mask           : (N,) bool — True at masked positions
    """
    N = seq_idx.shape[0]
    n_mask = max(1, int(N * mask_rate))
    perm = torch.randperm(N, device=seq_idx.device)[:n_mask]

    mask = torch.zeros(N, dtype=torch.bool, device=seq_idx.device)
    mask[perm] = True

    masked_scalar = x_scalar.clone()
    masked_scalar[mask] = 0.0

    masked_seq_idx = seq_idx.clone()
    masked_seq_idx[mask] = MASK_AA

    return masked_scalar, masked_seq_idx, mask


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


def sample_batch_mode(inverse_fold_prob: float) -> str:
    """
    Randomly choose the training mode for a batch.

    Returns "B" (inverse folding) with probability inverse_fold_prob,
    otherwise "A" (masked residue prediction).
    """
    return "B" if torch.rand(1).item() < inverse_fold_prob else "A"


class InverseFoldingLoss(nn.Module):
    """
    Cross-entropy loss over ALL residue positions (no masking).

    Used in Mode B batches: given full 3D structure, predict the complete
    amino acid sequence.

    Forward args:
        logits  : (N, 21) — raw predictions for all residues
        targets : (N,) int64 — true AA indices 0–20
        weights : (N,) float32 or None — per-residue loss weights (e.g. pLDDT/100).
                  Zero weight excludes a residue from the loss entirely.
                  If None, all residues contribute equally.

    Returns scalar loss.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                weights: torch.Tensor = None) -> torch.Tensor:
        if weights is None:
            return F.cross_entropy(logits, targets)

        # Weighted per-residue cross-entropy
        per_residue = F.cross_entropy(logits, targets, reduction="none")  # (N,)
        total_weight = weights.sum()
        if total_weight == 0:
            return logits.new_tensor(0.0)
        return (per_residue * weights).sum() / total_weight
