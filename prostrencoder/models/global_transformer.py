import warnings
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as ckpt

try:
    from flash_attn.modules.mha import MHA as FlashMHA
    from flash_attn.modules.mlp import GatedMlp
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    warnings.warn(
        "flash_attn not installed — falling back to torch.nn.MultiheadAttention. "
        "Install with: pip install flash-attn --no-build-isolation",
        stacklevel=2,
    )


class GlobalTransformerLayer(nn.Module):
    """
    Single pre-norm transformer layer.  Uses Flash Attention 2 when available
    (dao-ailab/flash-attention); falls back to torch.nn.MultiheadAttention.

    Args:
        d_model     : embedding dimension
        num_heads   : attention heads (must divide d_model)
        ff_mult     : feed-forward hidden = d_model * ff_mult
        dropout     : attention + FF dropout rate
        window_size : sliding window half-size per side; -1 = full attention
    """

    def __init__(self, d_model: int, num_heads: int, ff_mult: int = 4,
                 dropout: float = 0.1, window_size: int = -1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if HAS_FLASH_ATTN:
            self.attn = FlashMHA(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                causal=False,
                window_size=(window_size, window_size),
                rotary_emb_dim=d_model // num_heads,
            )
            self.ff = GatedMlp(
                in_features=d_model,
                hidden_features=d_model * ff_mult,
                activation=nn.functional.silu,
            )
            self._use_flash = True
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_model * ff_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * ff_mult, d_model),
                nn.Dropout(dropout),
            )
            self._use_flash = False

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x               : (B, N, d_model)
        key_padding_mask: (B, N) bool — True = padding position (ignored)
        Returns         : (B, N, d_model)
        """
        if self._use_flash:
            x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        else:
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed,
                                    key_padding_mask=key_padding_mask)
            x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class GlobalTransformer(nn.Module):
    """
    Stack of GlobalTransformerLayer blocks with an input bridge projection.

    Args:
        d_in                  : input feature dimension from GVP
        d_model               : transformer hidden dimension
        num_layers            : number of transformer layers
        num_heads             : attention heads
        ff_mult               : feed-forward multiplier
        dropout               : dropout rate
        window_size           : sliding window per side; -1 = full attention
        gradient_checkpointing: recompute activations on backward (saves memory)
    """

    def __init__(self, d_in: int, d_model: int, num_layers: int,
                 num_heads: int, ff_mult: int = 4, dropout: float = 0.1,
                 window_size: int = -1, gradient_checkpointing: bool = False):
        super().__init__()
        self.d_model = d_model
        self.gradient_checkpointing = gradient_checkpointing

        self.bridge = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([
            GlobalTransformerLayer(d_model, num_heads, ff_mult,
                                   dropout, window_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x               : (B, N, d_in)
        key_padding_mask: (B, N) bool — True = padding
        Returns         : (B, N, d_model)
        """
        x = self.bridge(x)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = ckpt(layer, x, key_padding_mask, use_reentrant=False)
            else:
                x = layer(x, key_padding_mask)
        return self.norm(x)
