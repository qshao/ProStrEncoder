# prostrencoder/models/encoder.py
import torch
import torch.nn as nn
from prostrencoder.models.gvp import GVP, GVPConv
from prostrencoder.models.walk_sampler import WalkSampler
from prostrencoder.models.walk_encoder import WalkEncoder
from prostrencoder.models.global_transformer import GlobalTransformer


class ProStrEncoder(nn.Module):
    """
    GVP-GNN local encoder + optional global transformer (v2).

    v1 behavior (tx_layers absent or 0): identical to original — GVP output
    directly projected to embedding space.

    v2 behavior (tx_layers > 0): GVP invariant output → bridge Linear →
    GlobalTransformer → embedding projection. Long-range context captured
    by attention; equivariance handled locally by GVP.

    Two-phase training support:
        self.use_transformer = False  →  Phase 1: bridge only, no attention layers
        self.use_transformer = True   →  Phase 2: full transformer

    The `use_transformer` flag is set by the Trainer at the phase transition.
    """

    def __init__(self, config: dict):
        super().__init__()
        s_n  = config["node_scalar_in"]
        v_n  = config["node_vector_in"]
        s_e  = config["edge_scalar_in"]
        v_e  = config["edge_vector_in"]
        s_h  = config["hidden_scalar"]
        v_h  = config["hidden_vector"]
        out  = config["output_dim"]
        L    = config["num_layers"]
        drop = config["dropout"]

        self._s_h = s_h
        self._v_h = v_h

        # ── Walk components ────────────────────────────────────────────────────
        self.walk_sampler = WalkSampler(
            num_walks=config["num_walks"],
            walk_length=config["walk_length"],
            max_deg_cap=config.get("max_deg_cap", 30),
        )
        walk_seq_len = config["walk_length"] + 2
        self.walk_encoder = WalkEncoder(
            vocab_size=23,
            embed_dim=config["walk_embed_dim"],
            num_heads=config["walk_heads"],
            num_layers=config["walk_layers"],
            ff_dim=config["walk_embed_dim"] * 4,
            max_seq_len=walk_seq_len,
            dropout=drop,
        )

        # ── GVP-GNN backbone ──────────────────────────────────────────────────
        self.node_in = GVP((s_n, v_n), (s_h, v_h))
        self.edge_in = GVP((s_e, v_e), (s_h // 4, v_e))
        s_e_h = s_h // 4
        self.layers = nn.ModuleList([
            GVPConv(node_in_dims=(s_h, v_h), edge_in_dims=(s_e_h, v_e),
                    node_out_dims=(s_h, v_h), drop_rate=drop)
            for _ in range(L)
        ])

        # ── Optional global transformer ───────────────────────────────────────
        tx_layers = config.get("tx_layers", 0)
        if tx_layers > 0:
            d_in = s_h + v_h   # GVP invariant dim (scalars + vector norms)
            self.transformer = GlobalTransformer(
                d_in=d_in,
                d_model=config["tx_d_model"],
                num_layers=tx_layers,
                num_heads=config["tx_num_heads"],
                ff_mult=config.get("tx_ff_mult", 4),
                dropout=config.get("tx_dropout", drop),
                window_size=config.get("attn_window_size", -1),
                gradient_checkpointing=config.get("gradient_checkpointing", False),
            )
            proj_in = config["tx_d_model"]
        else:
            self.transformer = None
            proj_in = s_h + v_h

        # Phase flag persisted as a bool buffer so it survives checkpoint save/load.
        # Trainer sets this via encoder.set_phase(2) at the phase transition.
        self.register_buffer("_use_transformer", torch.tensor(False))

        # ── Output projection ─────────────────────────────────────────────────
        self.out_proj = nn.Sequential(
            nn.LayerNorm(proj_in),
            nn.Linear(proj_in, out),
            nn.GELU(),
            nn.Linear(out, out),
        )

    @property
    def use_transformer(self) -> bool:
        return bool(self._use_transformer.item())

    @use_transformer.setter
    def use_transformer(self, value: bool) -> None:
        self._use_transformer.fill_(1 if value else 0)

    @property
    def hidden_dim(self) -> int:
        """Dimension of the hidden representation returned by forward(return_hidden=True)."""
        if self.transformer is not None:
            return self.transformer.d_model
        return self._s_h + self._v_h

    def _transformer_forward(self, gvp_out: torch.Tensor,
                             batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Packs flat node features into padded (B, max_N, D) tensors, runs the
        global transformer, then gathers back to flat (N, tx_d_model).

        gvp_out  : (N, s_h + v_h) — invariant GVP output
        batch_idx: (N,) int64     — protein index per node (from batch.batch)
        Returns  : (N, tx_d_model)
        """
        N = gvp_out.shape[0]
        B = int(batch_idx.max().item()) + 1
        device = gvp_out.device

        counts = torch.bincount(batch_idx, minlength=B)           # (B,)
        max_len = int(counts.max().item())

        cum = torch.zeros(B + 1, dtype=torch.long, device=device)
        cum[1:] = counts.cumsum(0)
        # batch_idx is guaranteed sorted by PyG Batch.from_data_list
        pos = torch.arange(N, device=device) - cum[batch_idx]     # within-protein pos

        # Scatter into padded tensor
        padded = gvp_out.new_zeros(B, max_len, gvp_out.shape[-1])
        padded[batch_idx, pos] = gvp_out

        # key_padding_mask: True = padding (ignored by attention)
        key_padding_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)
        key_padding_mask[batch_idx, pos] = False

        tx_out = self.transformer(padded, key_padding_mask=key_padding_mask)  # (B, max_len, d_model)
        return tx_out[batch_idx, pos]                              # gather back to (N, d_model)

    def forward(self, batch, return_hidden: bool = False):
        """
        batch        : PyG Batch or Data with fields seq_idx, x_scalar, x_vec,
                       edge_index, edge_scalar, edge_vec, and (for v2) batch.batch.
        return_hidden: if True, returns (output, hidden).
        """
        node_v     = batch.x_vec
        edge_index = batch.edge_index
        edge_s     = batch.edge_scalar
        edge_v     = batch.edge_vec

        # Walk embedding
        walk_tokens = self.walk_sampler(edge_index, batch.seq_idx)
        walk_emb    = self.walk_encoder(walk_tokens)
        node_s = torch.cat([batch.x_scalar, walk_emb], dim=-1)

        # GVP-GNN
        node_s, node_v = self.node_in(node_s, node_v)
        edge_s, edge_v = self.edge_in(edge_s, edge_v)
        for layer in self.layers:
            node_s, node_v = layer(node_s, node_v, edge_index, edge_s, edge_v)

        # Invariant GVP output
        v_norms = torch.linalg.vector_norm(node_v, dim=-1)    # (N, v_h)
        gvp_out = torch.cat([node_s, v_norms], dim=-1)        # (N, s_h + v_h)

        if self.transformer is not None:
            if self.use_transformer:
                # Phase 2: full transformer (bridge + attention layers + norm)
                batch_idx = getattr(batch, "batch", None)
                if batch_idx is None:
                    batch_idx = torch.zeros(node_s.shape[0], dtype=torch.long,
                                            device=node_s.device)
                hidden = self._transformer_forward(gvp_out, batch_idx)
            else:
                # Phase 1: bridge only — no attention, no extra compute
                hidden = self.transformer.bridge(gvp_out)
        else:
            hidden = gvp_out

        output = self.out_proj(hidden)
        if return_hidden:
            return output, hidden
        return output
