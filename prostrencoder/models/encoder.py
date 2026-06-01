import torch
import torch.nn as nn
from prostrencoder.models.gvp import GVP, GVPConv


class ProStrEncoder(nn.Module):
    """
    GVP-GNN protein structure encoder.

    Input : PyG Batch / Data with fields
              x_scalar   (N, 43)    scalar node features
              x_vec      (N, 3, 3)  vector node features (backbone frame)
              edge_index (2, E)
              edge_scalar (E, 16)
              edge_vec    (E, 1, 3)

    Output: (N, output_dim) rotation-invariant per-residue embeddings.
    """

    def __init__(self, config: dict):
        super().__init__()
        s_n  = config["node_scalar_in"]   # 43
        v_n  = config["node_vector_in"]   # 3
        s_e  = config["edge_scalar_in"]   # 16
        v_e  = config["edge_vector_in"]   # 1
        s_h  = config["hidden_scalar"]    # 128
        v_h  = config["hidden_vector"]    # 16
        out  = config["output_dim"]       # 512
        L    = config["num_layers"]       # 3
        drop = config["dropout"]          # 0.1

        # Input projections to hidden dims
        self.node_in = GVP((s_n, v_n), (s_h, v_h))
        self.edge_in = GVP((s_e, v_e), (s_h // 4, v_e))

        s_e_h = s_h // 4

        # Stack of GVPConv layers
        self.layers = nn.ModuleList([
            GVPConv(
                node_in_dims=(s_h, v_h),
                edge_in_dims=(s_e_h, v_e),
                node_out_dims=(s_h, v_h),
                drop_rate=drop,
            )
            for _ in range(L)
        ])

        # Output: combine scalar features with vector norms → linear
        self.out_proj = nn.Sequential(
            nn.LayerNorm(s_h + v_h),
            nn.Linear(s_h + v_h, out),
            nn.ReLU(),
            nn.Linear(out, out),
        )

    def forward(self, batch, return_hidden: bool = False):
        """
        batch : PyG Batch (or Data) object.
        return_hidden : if True, also return pre-projection features.
        Returns : (N, output_dim) tensor [, (N, s_h + v_h) hidden tensor].
        """
        node_s = batch.x_scalar          # (N, 43)
        node_v = batch.x_vec             # (N, 3, 3)
        edge_index = batch.edge_index    # (2, E)
        edge_s = batch.edge_scalar       # (E, 16)
        edge_v = batch.edge_vec          # (E, 1, 3)

        # Project to hidden dims
        node_s, node_v = self.node_in(node_s, node_v)
        edge_s, edge_v = self.edge_in(edge_s, edge_v)

        # GVPConv message passing
        for layer in self.layers:
            node_s, node_v = layer(node_s, node_v, edge_index, edge_s, edge_v)

        # Build rotation-invariant output: scalars + vector norms
        v_norms = torch.norm(node_v, dim=-1)  # (N, v_h)
        hidden = torch.cat([node_s, v_norms], dim=-1)  # (N, s_h + v_h)

        output = self.out_proj(hidden)
        if return_hidden:
            return output, hidden
        return output
