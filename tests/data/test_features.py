import numpy as np
import pytest
from prostrencoder.data.features import (
    aa_one_hot,
    compute_backbone_frame,
    compute_torsion_angles,
    rbf_encoding,
    compute_edge_directions,
)

N = 10

def make_linear_backbone(n=N, spacing=3.8):
    """Linear chain of residues: N, CA, C, O placed regularly."""
    bb = np.zeros((n, 4, 3), dtype=np.float32)
    for i in range(n):
        x = i * spacing
        bb[i, 0] = [x - 1.5, 0.0, 0.0]   # N
        bb[i, 1] = [x,       0.0, 0.0]   # CA
        bb[i, 2] = [x + 1.2, 0.4, 0.0]   # C
        bb[i, 3] = [x + 1.2, 1.6, 0.0]   # O
    return bb

def test_aa_one_hot_shape():
    seq_idx = np.array([0, 1, 20], dtype=np.int64)
    out = aa_one_hot(seq_idx)
    assert out.shape == (3, 21)
    assert out.dtype == np.float32

def test_aa_one_hot_sum_one():
    seq_idx = np.arange(21, dtype=np.int64)
    out = aa_one_hot(seq_idx)
    np.testing.assert_array_equal(out.sum(axis=1), np.ones(21))

def test_backbone_frame_shape():
    bb = make_linear_backbone()
    frame = compute_backbone_frame(bb)
    assert frame.shape == (N, 3, 3)
    assert frame.dtype == np.float32

def test_backbone_frame_orthonormal():
    bb = make_linear_backbone()
    frame = compute_backbone_frame(bb)
    for i in range(N):
        v1, v2, v3 = frame[i]
        np.testing.assert_allclose(np.linalg.norm(v1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(v2), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(v3), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.dot(v1, v2), 0.0, atol=1e-5)

def test_torsion_angles_shape():
    bb = make_linear_backbone()
    angles = compute_torsion_angles(bb)
    assert angles.shape == (N, 6)
    assert angles.dtype == np.float32

def test_torsion_angles_sin_cos_range():
    bb = make_linear_backbone()
    angles = compute_torsion_angles(bb)
    assert np.all(angles >= -1.0 - 1e-6)
    assert np.all(angles <=  1.0 + 1e-6)

def test_rbf_encoding_shape():
    dists = np.array([0.0, 5.0, 10.0, 20.0], dtype=np.float32)
    rbf = rbf_encoding(dists, num_rbf=16)
    assert rbf.shape == (4, 16)
    assert rbf.dtype == np.float32

def test_rbf_encoding_peaks_at_center():
    centers = np.linspace(0, 20, 16)
    for c in centers:
        dists = np.array([c], dtype=np.float32)
        rbf = rbf_encoding(dists, num_rbf=16)
        peak_idx = np.argmax(rbf[0])
        peak_center = centers[peak_idx]
        assert abs(peak_center - c) < 2.0

def test_edge_directions_unit_length():
    ca = np.array([[0,0,0],[3,4,0],[0,3,4]], dtype=np.float32)
    edge_index = np.array([[0,1],[1,2]], dtype=np.int64)
    edge_dist = np.array([5.0, np.sqrt(9+1+16)], dtype=np.float32)  # Actual Euclidean distances
    dirs = compute_edge_directions(ca, edge_index, edge_dist)
    assert dirs.shape == (2, 3)
    norms = np.linalg.norm(dirs, axis=1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)
