import numpy as np
import pytest
from prostrencoder.data.graph_builder import build_knn_graph

def linear_chain_coords(n, spacing=3.8):
    """n residues spaced spacing Å apart along x-axis."""
    return np.stack([np.arange(n) * spacing,
                     np.zeros(n),
                     np.zeros(n)], axis=1).astype(np.float32)

def test_knn_graph_shape_small():
    coords = linear_chain_coords(5)
    edge_index, edge_dist = build_knn_graph(coords, k=2)
    assert edge_index.shape[0] == 2
    assert edge_dist.shape[0] == edge_index.shape[1]
    assert edge_index.dtype == np.int64
    assert edge_dist.dtype == np.float32

def test_knn_graph_no_self_loops():
    coords = linear_chain_coords(10)
    edge_index, _ = build_knn_graph(coords, k=4)
    src, dst = edge_index
    assert not np.any(src == dst), "Graph must not contain self-loops"

def test_knn_graph_k_neighbors_per_node():
    n, k = 20, 5
    coords = linear_chain_coords(n)
    edge_index, _ = build_knn_graph(coords, k=k)
    src = edge_index[0]
    # Each non-boundary node should have exactly k outgoing edges;
    # boundary nodes may have fewer if k > (n-1).
    counts = np.bincount(src, minlength=n)
    assert np.all(counts <= k)
    # Interior nodes get exactly k neighbors
    assert np.all(counts[1:-1] == k)

def test_knn_graph_distances_are_positive():
    coords = linear_chain_coords(10)
    _, edge_dist = build_knn_graph(coords, k=3)
    assert np.all(edge_dist > 0)

def test_knn_graph_k_larger_than_n_does_not_crash():
    coords = linear_chain_coords(3)
    edge_index, edge_dist = build_knn_graph(coords, k=100)
    # Should not crash; each node gets at most n-1 = 2 neighbors
    assert edge_index.shape[1] == 3 * 2  # 3 nodes × 2 edges each (bidirectional)

def test_knn_graph_distances_match_euclidean():
    coords = linear_chain_coords(5, spacing=3.8)
    edge_index, edge_dist = build_knn_graph(coords, k=1)
    src, dst = edge_index
    for i in range(len(src)):
        expected = np.linalg.norm(coords[dst[i]] - coords[src[i]])
        assert abs(edge_dist[i] - expected) < 1e-4
