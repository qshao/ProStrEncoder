import numpy as np
from scipy.spatial import cKDTree


def build_knn_graph(ca_coords: np.ndarray, k: int = 30):
    """
    Build a directed k-NN graph from Cα coordinates.

    Residues with NaN coordinates are placed at a sentinel far-field
    location so they form no meaningful edges.

    Args:
        ca_coords : (N, 3) float32 array of Cα positions.
        k         : number of nearest neighbours per node (excluding self).

    Returns:
        edge_index : (2, E) int64 — [src_indices, dst_indices]
        edge_dist  : (E,) float32 — Euclidean distances in Å
    """
    n = len(ca_coords)
    coords = ca_coords.copy()
    nan_mask = np.isnan(coords).any(axis=1)
    coords[nan_mask] = 1e9  # sentinel: far from everything

    k_actual = min(k + 1, n)  # +1 because query includes self
    tree = cKDTree(coords)
    dists, indices = tree.query(coords, k=k_actual)
    # Ensure 2D even when n=1 or k_actual=1
    dists = np.atleast_2d(dists)
    indices = np.atleast_2d(indices)

    src_list, dst_list, dist_list = [], [], []
    for i in range(n):
        for j, d in zip(indices[i], dists[i]):
            if j != i:
                src_list.append(i)
                dst_list.append(int(j))
                dist_list.append(d)

    # Filter edges involving NaN residues
    if nan_mask.any():
        keep = [
            i for i, (s, d) in enumerate(zip(src_list, dst_list))
            if not nan_mask[s] and not nan_mask[d]
        ]
        src_list = [src_list[i] for i in keep]
        dst_list = [dst_list[i] for i in keep]
        dist_list = [dist_list[i] for i in keep]

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_dist = np.array(dist_list, dtype=np.float32)
    return edge_index, edge_dist
