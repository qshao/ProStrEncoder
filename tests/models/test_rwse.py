import numpy as np
import pytest
from prostrencoder.models.rwse import compute_rwse

def ring_graph(n):
    """Undirected ring graph: node i connects to (i-1) and (i+1) mod n."""
    src = list(range(n)) + list(range(n))
    dst = [(i + 1) % n for i in range(n)] + [(i - 1) % n for i in range(n)]
    return np.array([src, dst], dtype=np.int64)

def test_rwse_output_shape():
    edge_index = ring_graph(6)
    rwse = compute_rwse(edge_index, num_nodes=6, walk_length=8)
    assert rwse.shape == (6, 8)
    assert rwse.dtype == np.float32

def test_rwse_values_in_zero_one():
    edge_index = ring_graph(8)
    rwse = compute_rwse(edge_index, num_nodes=8, walk_length=8)
    assert np.all(rwse >= 0.0 - 1e-6)
    assert np.all(rwse <= 1.0 + 1e-6)

def test_rwse_symmetric_on_ring():
    """All nodes in a ring are equivalent; their RWSE should be identical."""
    n = 6
    edge_index = ring_graph(n)
    rwse = compute_rwse(edge_index, num_nodes=n, walk_length=8)
    for i in range(1, n):
        np.testing.assert_allclose(rwse[0], rwse[i], atol=1e-5)

def test_rwse_step1_is_zero_on_ring():
    """In a ring, a 1-step walk never returns to start (no self-loops)."""
    edge_index = ring_graph(6)
    rwse = compute_rwse(edge_index, num_nodes=6, walk_length=4)
    np.testing.assert_allclose(rwse[:, 0], np.zeros(6), atol=1e-6)

def test_rwse_step2_nonzero_on_ring():
    """In a 4-ring, a 2-step walk has 50% chance of returning."""
    edge_index = ring_graph(4)
    rwse = compute_rwse(edge_index, num_nodes=4, walk_length=4)
    np.testing.assert_allclose(rwse[:, 1], np.full(4, 0.5), atol=1e-5)

def test_rwse_disconnected_node():
    """An isolated node has RWSE = 0 for all steps."""
    # 3-node graph: 0-1-2 chain, node 3 is isolated
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    rwse = compute_rwse(edge_index, num_nodes=4, walk_length=4)
    np.testing.assert_allclose(rwse[3], np.zeros(4), atol=1e-6)
