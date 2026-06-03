import torch
import torch.nn as nn


class WalkSampler(nn.Module):
    """
    Samples random walks on a protein residue graph and returns AA-type
    token sequences for each residue.

    Not trainable — no parameters. Subclasses nn.Module for device consistency.

    For each residue i, draws num_walks independent random walks of walk_length
    steps on the k-NN graph defined by edge_index. Each walk is converted to
    a sequence of AA-type tokens and prepended with a CLS token.

    Args:
        num_walks  : K — number of independent walks per residue.
        walk_length: L — number of steps per walk (excluding starting node).

    Output token layout (length L + 2):
        [CLS(21), aa(i), aa(step1), aa(step2), ..., aa(stepL)]
    """

    CLS_TOKEN: int = 21

    def __init__(self, num_walks: int = 8, walk_length: int = 20,
                 max_deg_cap: int = 30):
        super().__init__()
        self.num_walks = num_walks
        self.walk_length = walk_length
        # Upper bound on node out-degree. Set to k_neighbors to avoid a
        # GPU→CPU sync (degree.max().item()) on every forward pass.
        self.max_deg_cap = max_deg_cap

    def forward(self, edge_index: torch.Tensor,
                seq_idx: torch.Tensor) -> torch.Tensor:
        """
        edge_index : (2, E) int64
        seq_idx    : (N,) int64 — AA type per residue (0–20)

        Returns: (N, num_walks, walk_length + 2) int64
        """
        N = seq_idx.shape[0]
        device = seq_idx.device
        src, dst = edge_index[0], edge_index[1]

        # ── Build padded adjacency tensor ─────────────────────────────────────
        degree = torch.zeros(N, dtype=torch.long, device=device)
        degree.scatter_add_(0, src, torch.ones_like(src))
        # Use the pre-configured cap (= k_neighbors) instead of degree.max().item()
        # to avoid stalling the GPU pipeline with a host-device sync every step.
        max_deg = max(self.max_deg_cap, 1)

        # Sort edges by source node so we can compute within-group slot indices
        order = src.argsort(stable=True)
        s_src, s_dst = src[order], dst[order]

        # cum_deg[i] = total number of edges with src < i (= start of node i's block)
        cum_deg = torch.zeros(N + 1, dtype=torch.long, device=device)
        cum_deg[1:] = degree.cumsum(0)
        slot = torch.arange(s_src.shape[0], device=device) - cum_deg[s_src]

        adj = torch.full((N, max_deg), -1, dtype=torch.long, device=device)
        adj[s_src, slot] = s_dst

        # ── Vectorised walk sampling ──────────────────────────────────────────
        # current[i, k] = node currently occupied by walk k of residue i
        current = (torch.arange(N, device=device)
                   .unsqueeze(1)
                   .expand(N, self.num_walks))                # (N, K)
        walk_nodes = [current]

        for _ in range(self.walk_length):
            deg = degree[current].clamp(min=1)               # (N, K)
            rand_slot = (torch.rand(N, self.num_walks, device=device)
                         * deg.float()).long().clamp(0, max_deg - 1)
            nxt = adj[current, rand_slot]                    # (N, K)
            # Stay in place when no valid neighbour (-1)
            nxt = torch.where(nxt >= 0, nxt, current)
            current = nxt
            walk_nodes.append(current)

        walk_nodes = torch.stack(walk_nodes, dim=2)          # (N, K, L+1)

        # ── Convert node indices → AA token IDs ──────────────────────────────
        walk_tokens = seq_idx[walk_nodes]                    # (N, K, L+1)

        # Prepend CLS
        cls = torch.full((N, self.num_walks, 1), self.CLS_TOKEN,
                         dtype=torch.long, device=device)
        return torch.cat([cls, walk_tokens], dim=2)          # (N, K, L+2)
