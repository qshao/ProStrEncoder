import torch
import torch.nn as nn
from prostrencoder.models.gvp import GVP, GVPConv
from prostrencoder.models.walk_sampler import WalkSampler
from prostrencoder.models.walk_encoder import WalkEncoder


class ProStrEncoder(nn.Module):
    """
    GVP-GNN protein structure encoder with CyberGFM walk-sequence embedding.

    Input (PyG Batch / Data):
        seq_idx    (N,) int64          — AA indices 0–20
        x_scalar   (N, 27) float32     — AA one-hot (21) + backbone torsion (6)
        x_vec      (N, 3, 3) float32   — local backbone frame (3 orthonormal vectors)
        edge_index (2, E) int64
        edge_scalar (E, 16) float32    — Gaussian RBF of Cα–Cα distances
        edge_vec   (E, 1, 3) float32   — unit direction vectors src→dst

    Forward pass:
        1. WalkSampler samples K walks per residue → token seqs (N, K, L+2)
        2. WalkEncoder encodes token seqs → walk_emb (N, walk_embed_dim)
        3. node_s = cat([x_scalar, walk_emb], dim=-1)  → (N, 91)
        4. GVP-GNN processes node_s and node_v through L layers
        5. Output: scalars + vector norms → projection → (N, output_dim)

    Output: (N, output_dim) rotation-invariant per-residue embeddings.
    The walk embedding is invariant (derived from AA types, not 3D coords).
    The GVP output is invariant by construction. Combined output is invariant.
    """

    def __init__(self, config: dict):
        super().__init__()
        s_n  = config["node_scalar_in"]    # 91 = 27 + walk_embed_dim
        v_n  = config["node_vector_in"]    # 3
        s_e  = config["edge_scalar_in"]    # 16
        v_e  = config["edge_vector_in"]    # 1
        s_h  = config["hidden_scalar"]     # 128
        v_h  = config["hidden_vector"]     # 16
        out  = config["output_dim"]        # 512
        L    = config["num_layers"]        # 3
        drop = config["dropout"]           # 0.1

        # ── CyberGFM-inspired walk components ────────────────────────────────
        self.walk_sampler = WalkSampler(
            num_walks=config["num_walks"],      # 8
            walk_length=config["walk_length"],  # 20
        )
        walk_seq_len = config["walk_length"] + 2   # CLS + walk_length+1 nodes
        self.walk_encoder = WalkEncoder(
            vocab_size=22,
            embed_dim=config["walk_embed_dim"],         # 64
            num_heads=config["walk_heads"],             # 4
            num_layers=config["walk_layers"],           # 4
            ff_dim=config["walk_embed_dim"] * 4,        # 256
            max_seq_len=walk_seq_len,
            dropout=drop,
        )

        # ── GVP-GNN backbone (unchanged) ─────────────────────────────────────
        self.node_in = GVP((s_n, v_n), (s_h, v_h))
        self.edge_in = GVP((s_e, v_e), (s_h // 4, v_e))

        s_e_h = s_h // 4
        self.layers = nn.ModuleList([
            GVPConv(
                node_in_dims=(s_h, v_h),
                edge_in_dims=(s_e_h, v_e),
                node_out_dims=(s_h, v_h),
                drop_rate=drop,
            )
            for _ in range(L)
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(s_h + v_h),
            nn.Linear(s_h + v_h, out),
            nn.ReLU(),
            nn.Linear(out, out),
        )

    def forward(self, batch, return_hidden: bool = False):
        """
        batch        : PyG Batch or Data.
        return_hidden: if True, returns (output, hidden) where
                       hidden is (N, hidden_scalar + hidden_vector).
        """
        node_v     = batch.x_vec          # (N, 3, 3)
        edge_index = batch.edge_index     # (2, E)
        edge_s     = batch.edge_scalar    # (E, 16)
        edge_v     = batch.edge_vec       # (E, 1, 3)

        # ── Walk-sequence embedding (CyberGFM-inspired) ───────────────────────
        walk_tokens = self.walk_sampler(edge_index, batch.seq_idx)  # (N, K, L+2)
        walk_emb    = self.walk_encoder(walk_tokens)                 # (N, 64)

        # Concatenate with scalar node features
        node_s = torch.cat([batch.x_scalar, walk_emb], dim=-1)      # (N, 91)

        # ── GVP-GNN message passing ───────────────────────────────────────────
        node_s, node_v = self.node_in(node_s, node_v)
        edge_s, edge_v = self.edge_in(edge_s, edge_v)

        for layer in self.layers:
            node_s, node_v = layer(node_s, node_v, edge_index, edge_s, edge_v)

        # Rotation-invariant output: scalar features + vector norms
        v_norms = torch.norm(node_v, dim=-1)           # (N, v_h)
        hidden  = torch.cat([node_s, v_norms], dim=-1) # (N, s_h + v_h)

        output = self.out_proj(hidden)
        if return_hidden:
            return output, hidden
        return output
