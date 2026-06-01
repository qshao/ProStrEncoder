import torch
import torch.nn as nn
import torch.nn.functional as F


class GVP(nn.Module):
    """
    Geometric Vector Perceptron (Jing et al. 2021).

    Transforms (scalar_feats, vector_feats) → (scalar_out, vector_out)
    while preserving E(3) equivariance of the vector branch.

    in_dims  = (s_in, v_in)   where s_in is scalar dim, v_in is # of 3D vectors
    out_dims = (s_out, v_out)
    """

    def __init__(self, in_dims: tuple, out_dims: tuple,
                 scalar_act=nn.ReLU(), vector_gate: bool = True):
        super().__init__()
        s_in, v_in = in_dims
        s_out, v_out = out_dims

        self.v_in  = v_in
        self.v_out = v_out
        self.vector_gate = vector_gate

        # v_h: hidden vector dim used to squeeze vector info into scalars
        v_h = max(v_in, v_out) if v_in > 0 or v_out > 0 else 0

        self.W_h = nn.Linear(v_in, v_h, bias=False) if v_in > 0 and v_h > 0 else None
        self.W_V = nn.Linear(v_in, v_out, bias=False) if v_in > 0 and v_out > 0 else None

        # scalar projection: s_in + v_h norms → s_out
        self.W_s = nn.Linear(s_in + (v_h if v_in > 0 else 0), s_out)

        if vector_gate and v_out > 0:
            self.gate_linear = nn.Linear(s_out, v_out)
        else:
            self.gate_linear = None

        self.scalar_act = scalar_act

    @staticmethod
    def _vec_linear(W: nn.Linear, V: torch.Tensor) -> torch.Tensor:
        """Apply W to the vector-channel dimension of V (..., v_in, 3).

        W.weight has shape (v_out, v_in).  We contract over the v_in axis
        (index v) while leaving the 3D spatial axis (index x) untouched,
        producing (..., v_out, 3).  This makes the transform equivariant
        because rotation acts only on the spatial axis x.
        """
        return torch.einsum("...vx,ov->...ox", V, W.weight)

    def forward(self, s: torch.Tensor, V: torch.Tensor):
        """
        s : (..., s_in)
        V : (..., v_in, 3)
        Returns (s_out, V_out) with shapes (..., s_out) and (..., v_out, 3).
        """
        # ── Vector → scalar (via norms) ───────────────────────────────────────
        if self.v_in > 0 and self.W_h is not None:
            V_h = self._vec_linear(self.W_h, V)          # (..., v_h, 3)
            norms = torch.norm(V_h, dim=-1)               # (..., v_h)
            s_cat = torch.cat([s, norms], dim=-1)
        else:
            s_cat = s

        # ── Scalar branch ─────────────────────────────────────────────────────
        s_out = self.W_s(s_cat)
        if self.scalar_act is not None:
            s_out = self.scalar_act(s_out)

        # ── Vector branch ─────────────────────────────────────────────────────
        if self.v_out == 0 or self.v_in == 0 or self.W_V is None:
            V_out = V.new_zeros(*V.shape[:-2], self.v_out, 3)
        else:
            V_out = self._vec_linear(self.W_V, V)         # (..., v_out, 3)
            if self.vector_gate and self.gate_linear is not None:
                gates = torch.sigmoid(self.gate_linear(s_out))  # (..., v_out)
                V_out = gates.unsqueeze(-1) * V_out

        return s_out, V_out


class GVPConv(nn.Module):
    """
    One round of GVP-based message passing.

    Messages are computed by concatenating source-node and edge features and
    passing them through two GVP layers.  Messages are sum-aggregated at
    destination nodes.  Node states are updated by concatenating the
    destination-node state with the aggregated messages and passing through
    two more GVP layers, with a residual connection on the scalar branch.
    """

    def __init__(self, node_in_dims: tuple, edge_in_dims: tuple,
                 node_out_dims: tuple, drop_rate: float = 0.1):
        super().__init__()
        s_n, v_n = node_in_dims
        s_e, v_e = edge_in_dims
        s_o, v_o = node_out_dims

        # Message network: (src_node ∥ edge) → message
        self.msg1 = GVP((s_n + s_e, v_n + v_e), (s_o, v_o))
        self.msg2 = GVP((s_o, v_o), (s_o, v_o), scalar_act=None, vector_gate=False)

        # Update network: (dst_node ∥ agg_message) → new node state
        self.upd1 = GVP((s_n + s_o, v_n + v_o), (s_o, v_o))
        self.upd2 = GVP((s_o, v_o), (s_o, v_o), scalar_act=None, vector_gate=False)

        self.norm_s = nn.LayerNorm(s_o)
        self.drop   = nn.Dropout(drop_rate)

        # Scalar residual projection when dims differ
        self.res_proj = nn.Linear(s_n, s_o, bias=False) if s_n != s_o else nn.Identity()

    def forward(self, node_s, node_v, edge_index, edge_s, edge_v):
        """
        node_s : (N, s_n)    node_v : (N, v_n, 3)
        edge_s : (E, s_e)    edge_v : (E, v_e, 3)
        edge_index : (2, E)
        Returns updated (node_s, node_v) with dims (s_o, v_o).
        """
        src, dst = edge_index[0], edge_index[1]
        N = node_s.shape[0]
        E = src.shape[0]  # actual number of edges from edge_index

        # Clip edge features to match the number of edges in edge_index.
        # This handles cases where edge feature tensors are pre-allocated
        # with a capacity larger than the actual edge count.
        edge_s = edge_s[:E]
        edge_v = edge_v[:E]

        # ── Build messages ────────────────────────────────────────────────────
        m_s = torch.cat([node_s[src], edge_s], dim=-1)          # (E, s_n+s_e)
        m_v = torch.cat([node_v[src], edge_v], dim=-2)          # (E, v_n+v_e, 3)

        m_s, m_v = self.msg1(m_s, m_v)
        m_s, m_v = self.msg2(m_s, m_v)

        # ── Aggregate messages (sum) ──────────────────────────────────────────
        agg_s = node_s.new_zeros(N, m_s.shape[-1])
        agg_s.scatter_add_(0, dst.unsqueeze(-1).expand_as(m_s), m_s)

        agg_v = node_v.new_zeros(N, m_v.shape[-2], 3)
        dst_idx = dst.view(-1, 1, 1).expand_as(m_v)
        agg_v.scatter_add_(0, dst_idx, m_v)

        # ── Update node states ────────────────────────────────────────────────
        upd_s = torch.cat([node_s, agg_s], dim=-1)              # (N, s_n+s_o)
        upd_v = torch.cat([node_v, agg_v], dim=-2)              # (N, v_n+v_o, 3)

        new_s, new_v = self.upd1(upd_s, upd_v)
        new_s, new_v = self.upd2(new_s, new_v)

        new_s = self.norm_s(self.drop(new_s) + self.res_proj(node_s))

        return new_s, new_v
