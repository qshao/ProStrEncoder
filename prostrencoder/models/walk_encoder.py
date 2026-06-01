import torch
import torch.nn as nn


class WalkEncoder(nn.Module):
    """
    Encodes random-walk token sequences into per-residue scalar embeddings.

    Inspired by CyberGFM's walk-tokenizer + BERT design (cybermonic/CyberGFM).

    For each residue, K walk sequences are encoded independently by a
    transformer encoder. The CLS-token hidden state from each walk is
    extracted and the K states are mean-pooled to produce one embedding
    per residue.

    The output is rotation-invariant because it depends only on AA-type
    token IDs (integers), never on 3D coordinates.

    Args:
        vocab_size  : number of distinct token IDs (default 22: 0-20 AA + 21 CLS).
        embed_dim   : transformer hidden dimension and output dimension.
        num_heads   : attention heads (embed_dim must be divisible by num_heads).
        num_layers  : number of transformer encoder layers.
        ff_dim      : feed-forward hidden dimension inside each transformer layer.
        max_seq_len : maximum sequence length (must be >= walk_length + 2).
        dropout     : dropout rate applied inside transformer layers.
    """

    def __init__(self, vocab_size: int = 22, embed_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 4,
                 ff_dim: int = 256, max_seq_len: int = 24,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed   = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # pre-norm: more stable for shallow transformers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, walk_tokens: torch.Tensor) -> torch.Tensor:
        """
        walk_tokens : (N, K, L) int64
            N = number of residues, K = num_walks, L = walk_length + 2
            Position 0 of each sequence must be the CLS token (index 21).

        Returns: (N, embed_dim) float32 — one embedding per residue.
        """
        N, K, L = walk_tokens.shape
        device = walk_tokens.device

        # Flatten (N, K) into a single batch dimension: (N*K, L)
        tokens = walk_tokens.reshape(N * K, L)

        # Token + positional embeddings
        pos = torch.arange(L, device=device).unsqueeze(0)   # (1, L)
        x = self.token_embed(tokens) + self.pos_embed(pos)  # (N*K, L, D)

        # Transformer encoding
        x = self.transformer(x)    # (N*K, L, D)
        x = self.norm(x)

        # Extract CLS hidden state (position 0)
        cls_out = x[:, 0, :]                   # (N*K, D)

        # Mean-pool K walks per residue
        cls_out = cls_out.reshape(N, K, -1)    # (N, K, D)
        return cls_out.mean(dim=1)             # (N, D)
