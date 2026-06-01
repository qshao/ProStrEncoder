import numpy as np
from scipy.sparse import csr_matrix, eye as speye


def compute_rwse(edge_index: np.ndarray, num_nodes: int,
                 walk_length: int = 16) -> np.ndarray:
    """
    Compute Random Walk Structural Encoding (RWSE).

    RWSE[i, k] = probability of returning to node i after exactly k+1 steps,
    starting from node i, using the row-stochastic random walk matrix
    P = D^{-1} A (A = adjacency, D = out-degree diagonal).

    Isolated nodes (degree 0) get RWSE = 0.

    Args:
        edge_index  : (2, E) int64 — directed edge list [src, dst]
        num_nodes   : N
        walk_length : K — number of steps to compute

    Returns:
        rwse : (N, K) float32
    """
    N = num_nodes
    src = edge_index[0]
    dst = edge_index[1]

    data = np.ones(len(src), dtype=np.float64)
    A = csr_matrix((data, (src, dst)), shape=(N, N))

    # Row-stochastic P = D^{-1} A
    deg = np.array(A.sum(axis=1), dtype=np.float64).flatten()
    inv_deg = np.divide(1.0, deg, out=np.zeros_like(deg), where=(deg > 0))
    D_inv = csr_matrix(
        (inv_deg, (np.arange(N), np.arange(N))), shape=(N, N)
    )
    P = D_inv @ A

    rwse = np.zeros((N, walk_length), dtype=np.float32)
    P_k = speye(N, format="csr", dtype=np.float64)  # P^0 = I

    for k in range(walk_length):
        P_k = P_k @ P
        rwse[:, k] = P_k.diagonal().astype(np.float32)

    return rwse
