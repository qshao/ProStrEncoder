import torch
import pytest
from prostrencoder.training.objectives import MaskedResidueLoss, apply_masking

def test_apply_masking_mask_rate():
    """~15% of residues should be masked."""
    N = 1000
    seq_idx = torch.randint(0, 20, (N,))
    masked_scalar, mask = apply_masking(seq_idx, torch.randn(N, 27), mask_rate=0.15)
    assert mask.dtype == torch.bool
    assert mask.shape == (N,)
    frac = mask.float().mean().item()
    assert 0.10 < frac < 0.20, f"Mask rate {frac:.3f} out of expected 10-20% range"

def test_apply_masking_zeros_masked_positions():
    """Masked positions in x_scalar should be zeroed out."""
    N = 50
    seq_idx = torch.randint(0, 20, (N,))
    x_scalar = torch.ones(N, 27)
    masked_scalar, mask = apply_masking(seq_idx, x_scalar, mask_rate=0.5)
    # Masked positions (mask=True) should have x_scalar = 0
    assert torch.all(masked_scalar[mask] == 0.0)
    # Unmasked positions should be unchanged
    assert torch.all(masked_scalar[~mask] == 1.0)

def test_apply_masking_returns_bool_tensor():
    seq_idx = torch.randint(0, 20, (20,))
    _, mask = apply_masking(seq_idx, torch.randn(20, 27))
    assert mask.dtype == torch.bool

def test_masked_residue_loss_shape():
    N = 30
    loss_fn = MaskedResidueLoss()
    logits = torch.randn(N, 21)
    targets = torch.randint(0, 21, (N,))
    mask = torch.zeros(N, dtype=torch.bool)
    mask[:5] = True
    loss = loss_fn(logits, targets, mask)
    assert loss.ndim == 0  # scalar

def test_masked_residue_loss_only_uses_masked():
    """Loss should only depend on masked positions."""
    N = 20
    loss_fn = MaskedResidueLoss()
    logits = torch.randn(N, 21)
    targets = torch.randint(0, 21, (N,))
    mask = torch.zeros(N, dtype=torch.bool)
    mask[:3] = True

    loss1 = loss_fn(logits, targets, mask)

    # Perturb unmasked logits — loss must not change
    logits2 = logits.clone()
    logits2[~mask] += 100.0
    loss2 = loss_fn(logits2, targets, mask)

    torch.testing.assert_close(loss1, loss2)

def test_masked_residue_loss_no_masked_returns_zero():
    N = 20
    loss_fn = MaskedResidueLoss()
    logits = torch.randn(N, 21)
    targets = torch.randint(0, 21, (N,))
    mask = torch.zeros(N, dtype=torch.bool)  # nothing masked
    loss = loss_fn(logits, targets, mask)
    assert loss.item() == 0.0

# ── Training gradient smoke test ──────────────────────────────────────────────

from torch_geometric.data import Data, Batch
from prostrencoder.models.encoder import ProStrEncoder
from prostrencoder.models.gnn import ResidueHead

def _fake_batch(n=30, e=60):
    data = Data(
        seq_idx=torch.randint(0, 20, (n,)),
        x_scalar=torch.randn(n, 27),
        x_vec=torch.randn(n, 3, 3),
        edge_index=torch.randint(0, n, (2, e)),
        edge_scalar=torch.randn(e, 16),
        edge_vec=torch.randn(e, 1, 3),
    )
    return Batch.from_data_list([data])

def test_training_step_computes_gradient():
    config = {
        "num_layers": 1,
        "node_scalar_in": 91, "node_vector_in": 3,
        "edge_scalar_in": 16, "edge_vector_in": 1,
        "hidden_scalar": 32, "hidden_vector": 4,
        "output_dim": 64, "dropout": 0.0,
        "num_walks": 8,
        "walk_length": 20,
        "walk_embed_dim": 64,
        "walk_layers": 4,
        "walk_heads": 4,
    }
    encoder = ProStrEncoder(config)
    head = ResidueHead(in_dim=32 + 4)
    loss_fn = MaskedResidueLoss()

    batch = _fake_batch()
    masked_scalar, mask = apply_masking(batch.seq_idx, batch.x_scalar)
    batch.x_scalar = masked_scalar

    _, hidden = encoder(batch, return_hidden=True)
    logits = head(hidden)
    loss = loss_fn(logits, batch.seq_idx, mask)
    loss.backward()

    for p in encoder.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()
