import torch
import pytest
from prostrencoder.models.gvp import GVP, GVPConv

def rand_node_features(n, s_dim, v_dim):
    return torch.randn(n, s_dim), torch.randn(n, v_dim, 3)

def rand_edge_features(e, s_dim, v_dim):
    return torch.randn(e, s_dim), torch.randn(e, v_dim, 3)

# ── GVP tests ─────────────────────────────────────────────────────────────────

def test_gvp_output_shapes():
    gvp = GVP(in_dims=(27, 3), out_dims=(64, 8))
    s, V = rand_node_features(10, 27, 3)
    s_out, V_out = gvp(s, V)
    assert s_out.shape == (10, 64)
    assert V_out.shape == (10, 8, 3)

def test_gvp_no_vector_out():
    gvp = GVP(in_dims=(16, 4), out_dims=(32, 0))
    s, V = rand_node_features(5, 16, 4)
    s_out, V_out = gvp(s, V)
    assert s_out.shape == (5, 32)
    assert V_out.shape == (5, 0, 3)

def test_gvp_scalar_only_input():
    gvp = GVP(in_dims=(16, 0), out_dims=(32, 4))
    s = torch.randn(5, 16)
    V = torch.zeros(5, 0, 3)
    s_out, V_out = gvp(s, V)
    assert s_out.shape == (5, 32)
    assert V_out.shape == (5, 4, 3)

def test_gvp_equivariant_vectors():
    """Rotating input vectors by R should rotate output vectors by R."""
    gvp = GVP(in_dims=(8, 4), out_dims=(8, 4))
    gvp.eval()
    s = torch.randn(3, 8)
    V = torch.randn(3, 4, 3)

    # Random rotation matrix
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    V_rot = (Q @ V.transpose(-1, -2)).transpose(-1, -2)

    with torch.no_grad():
        s_out, V_out = gvp(s, V)
        s_out_rot, V_out_rot = gvp(s, V_rot)

    torch.testing.assert_close(s_out, s_out_rot, atol=1e-5, rtol=1e-4)
    expected_V_out_rot = (Q @ V_out.transpose(-1, -2)).transpose(-1, -2)
    torch.testing.assert_close(V_out_rot, expected_V_out_rot, atol=1e-5, rtol=1e-4)

# ── GVPConv tests ──────────────────────────────────────────────────────────────

def make_chain_graph(n=6):
    """Bidirectional chain: 0↔1↔2↔…↔(n-1)."""
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    return torch.tensor([src, dst], dtype=torch.long)

def test_gvpconv_output_shapes():
    conv = GVPConv(
        node_in_dims=(43, 3),
        edge_in_dims=(16, 1),
        node_out_dims=(128, 16),
    )
    N = 10
    edge_index = make_chain_graph(N)        # 18 edges for N=10
    E = edge_index.shape[1]
    node_s, node_v = rand_node_features(N, 43, 3)
    edge_s, edge_v = rand_edge_features(E, 16, 1)
    out_s, out_v = conv(node_s, node_v, edge_index, edge_s, edge_v)
    assert out_s.shape == (N, 128)
    assert out_v.shape == (N, 16, 3)

def test_gvpconv_equivariant():
    """Rotating all 3D vectors by R should rotate output vectors by R."""
    conv = GVPConv(
        node_in_dims=(8, 2), edge_in_dims=(4, 1), node_out_dims=(8, 2),
    )
    conv.eval()
    N, E = 6, 8
    node_s, node_v = rand_node_features(N, 8, 2)
    edge_s, edge_v = rand_edge_features(E, 4, 1)
    edge_index = torch.randint(0, N, (2, E))

    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    node_v_rot = (Q @ node_v.transpose(-1, -2)).transpose(-1, -2)
    edge_v_rot = (Q @ edge_v.transpose(-1, -2)).transpose(-1, -2)

    with torch.no_grad():
        out_s, out_v = conv(node_s, node_v, edge_index, edge_s, edge_v)
        out_s_rot, out_v_rot = conv(node_s, node_v_rot, edge_index, edge_s, edge_v_rot)

    torch.testing.assert_close(out_s, out_s_rot, atol=1e-4, rtol=1e-3)
    expected = (Q @ out_v.transpose(-1, -2)).transpose(-1, -2)
    torch.testing.assert_close(out_v_rot, expected, atol=1e-4, rtol=1e-3)
