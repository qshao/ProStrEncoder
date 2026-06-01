import torch
import pytest
from prostrencoder.models.walk_encoder import WalkEncoder


# Small encoder for fast tests
ENC = WalkEncoder(
    vocab_size=22, embed_dim=64, num_heads=4,
    num_layers=2, ff_dim=128, max_seq_len=24,
)


def test_output_shape():
    N, K, L = 10, 4, 7
    tokens = torch.randint(0, 22, (N, K, L))
    out = ENC(tokens)
    assert out.shape == (N, 64)
    assert out.dtype == torch.float32


def test_output_is_finite():
    tokens = torch.randint(0, 22, (8, 4, 7))
    out = ENC(tokens)
    assert torch.all(torch.isfinite(out))


def test_different_walks_change_output():
    """Changing one residue's walk tokens must change only that residue's embedding."""
    ENC.eval()
    tokens1 = torch.randint(0, 21, (6, 4, 7))
    tokens2 = tokens1.clone()
    tokens2[0, :, 3] = (tokens2[0, :, 3] + 5) % 21   # mutate residue 0
    with torch.no_grad():
        out1 = ENC(tokens1)
        out2 = ENC(tokens2)
    assert not torch.allclose(out1[0], out2[0])        # residue 0 changed
    torch.testing.assert_close(out1[1:], out2[1:])     # residues 1‥5 unchanged


def test_single_walk_valid():
    """K=1 must still produce correct (N, embed_dim) output."""
    enc = WalkEncoder(vocab_size=22, embed_dim=32, num_heads=2,
                      num_layers=1, ff_dim=64, max_seq_len=24)
    tokens = torch.randint(0, 22, (5, 1, 7))
    out = enc(tokens)
    assert out.shape == (5, 32)


def test_deterministic_in_eval_mode():
    """eval() + no dropout → identical outputs for same input."""
    ENC.eval()
    tokens = torch.randint(0, 22, (6, 4, 7))
    with torch.no_grad():
        out1 = ENC(tokens)
        out2 = ENC(tokens)
    torch.testing.assert_close(out1, out2)


def test_cls_position_matters():
    """Replacing the non-CLS tokens with random values should change output."""
    ENC.eval()
    N, K, L = 4, 4, 7
    tokens_a = torch.randint(0, 21, (N, K, L))
    tokens_a[:, :, 0] = 21   # CLS at position 0
    tokens_b = tokens_a.clone()
    tokens_b[:, :, 1:] = torch.randint(0, 21, (N, K, L - 1))
    with torch.no_grad():
        out_a = ENC(tokens_a)
        out_b = ENC(tokens_b)
    assert not torch.allclose(out_a, out_b)
