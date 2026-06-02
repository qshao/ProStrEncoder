import pytest
import torch

try:
    import flash_attn  # noqa: F401
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def _make_layer(d_model=64, num_heads=4, window_size=-1):
    from prostrencoder.models.global_transformer import GlobalTransformerLayer
    return GlobalTransformerLayer(d_model=d_model, num_heads=num_heads,
                                  ff_mult=4, dropout=0.0,
                                  window_size=window_size)


def _make_transformer(d_in=144, d_model=64, num_layers=2, num_heads=4):
    from prostrencoder.models.global_transformer import GlobalTransformer
    return GlobalTransformer(d_in=d_in, d_model=d_model, num_layers=num_layers,
                             num_heads=num_heads, ff_mult=4, dropout=0.0)


def test_layer_output_shape():
    layer = _make_layer()
    x = torch.randn(2, 10, 64)   # (B, N, D)
    out = layer(x)
    assert out.shape == (2, 10, 64)


def test_layer_with_padding_mask():
    layer = _make_layer()
    x = torch.randn(2, 10, 64)
    # Last 3 positions of second sequence are padding
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[1, 7:] = True
    out = layer(x, key_padding_mask=mask)
    assert out.shape == (2, 10, 64)
    assert torch.isfinite(out).all()


def test_transformer_bridges_input_dim():
    """Bridge must project d_in -> d_model correctly."""
    model = _make_transformer(d_in=144, d_model=64)
    x = torch.randn(2, 12, 144)   # (B, N, d_in)
    out = model(x)
    assert out.shape == (2, 12, 64)


def test_transformer_output_finite():
    model = _make_transformer()
    x = torch.randn(3, 8, 144)
    out = model(x)
    assert torch.isfinite(out).all()


def test_transformer_gradient_checkpointing():
    from prostrencoder.models.global_transformer import GlobalTransformer
    model = GlobalTransformer(d_in=144, d_model=64, num_layers=2,
                              num_heads=4, ff_mult=4, dropout=0.0,
                              gradient_checkpointing=True)
    x = torch.randn(2, 8, 144, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None


@pytest.mark.skipif(not HAS_FLASH_ATTN, reason="flash_attn not installed")
def test_flash_attn_layer_output_shape():
    """Flash and fallback layers produce the same output shape."""
    from prostrencoder.models.global_transformer import GlobalTransformerLayer
    x = torch.randn(2, 16, 64)
    layer = GlobalTransformerLayer(64, 4, dropout=0.0)
    out = layer(x)
    assert out.shape == (2, 16, 64)
