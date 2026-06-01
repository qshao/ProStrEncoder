import torch
import pytest
from torch_geometric.data import Data, Batch
from prostrencoder.models.encoder import ProStrEncoder


def make_fake_batch(n_proteins=2, n_residues=20, n_edges=60):
    data_list = []
    for _ in range(n_proteins):
        N, E = n_residues, n_edges
        data = Data(
            seq_idx=torch.randint(0, 21, (N,)),
            x_scalar=torch.randn(N, 27),      # 21 AA one-hot + 6 torsion (no RWSE)
            x_vec=torch.randn(N, 3, 3),
            edge_index=torch.randint(0, N, (2, E)),
            edge_scalar=torch.randn(E, 16),
            edge_vec=torch.randn(E, 1, 3),
        )
        data_list.append(data)
    return Batch.from_data_list(data_list)


CONFIG = {
    "num_layers": 2,
    "node_scalar_in": 91,      # 27 (aa+torsion) + 64 (walk_embed_dim)
    "node_vector_in": 3,
    "edge_scalar_in": 16,
    "edge_vector_in": 1,
    "hidden_scalar": 64,
    "hidden_vector": 8,
    "output_dim": 128,
    "dropout": 0.0,
    # Walk config (small values for test speed)
    "num_walks": 4,
    "walk_length": 5,
    "walk_embed_dim": 64,
    "walk_layers": 2,
    "walk_heads": 4,
}


def test_encoder_output_shape():
    model = ProStrEncoder(CONFIG)
    batch = make_fake_batch()
    total_N = batch.x_scalar.shape[0]
    emb = model(batch)
    assert emb.shape == (total_N, 128)


def test_encoder_output_is_finite():
    model = ProStrEncoder(CONFIG)
    batch = make_fake_batch()
    emb = model(batch)
    assert torch.all(torch.isfinite(emb))


def test_encoder_equivariant_embedding():
    """Embedding must not change when 3D vectors are rotated (rotation-invariant)."""
    model = ProStrEncoder(CONFIG)
    model.eval()
    batch = make_fake_batch(n_proteins=1)

    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    batch_rot = batch.clone()
    batch_rot.x_vec    = (Q @ batch.x_vec.transpose(-1, -2)).transpose(-1, -2)
    batch_rot.edge_vec = (Q @ batch.edge_vec.transpose(-1, -2)).transpose(-1, -2)

    with torch.no_grad():
        torch.manual_seed(0)
        emb     = model(batch)
        torch.manual_seed(0)
        emb_rot = model(batch_rot)

    # Walk embedding uses seq_idx (AA types) not 3D vectors → invariant
    # GVP output is invariant → total output invariant
    torch.testing.assert_close(emb, emb_rot, atol=1e-4, rtol=1e-3)


def test_encoder_parameter_count():
    model = ProStrEncoder(CONFIG)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 10_000, "Model seems too small"
    assert n_params < 100_000_000, "Model seems too large for config"


def test_encoder_return_hidden():
    """return_hidden=True must return (output, hidden) where hidden has shape (N, s_h+v_h)."""
    model = ProStrEncoder(CONFIG)
    batch = make_fake_batch(n_proteins=1)
    total_N = batch.x_scalar.shape[0]
    with torch.no_grad():
        out, hidden = model(batch, return_hidden=True)
    assert out.shape == (total_N, 128)
    s_h = CONFIG["hidden_scalar"]
    v_h = CONFIG["hidden_vector"]
    assert hidden.shape == (total_N, s_h + v_h)
