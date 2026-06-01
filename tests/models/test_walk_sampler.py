import torch
from prostrencoder.models.walk_sampler import WalkSampler


def chain_graph(n):
    """Bidirectional chain 0↔1↔2↔…↔(n-1)."""
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    return torch.tensor([src, dst], dtype=torch.long)


SAMPLER = WalkSampler(num_walks=4, walk_length=5)


def test_output_shape():
    n = 10
    edge_index = chain_graph(n)
    seq_idx = torch.randint(0, 21, (n,))
    out = SAMPLER(edge_index, seq_idx)
    # seq = [CLS, start_aa, step1_aa, ..., step5_aa] → length 7
    assert out.shape == (n, 4, 7)   # (N, num_walks, walk_length + 2)
    assert out.dtype == torch.long


def test_first_token_is_cls():
    """Every walk must start with the CLS token (index 21)."""
    n = 8
    edge_index = chain_graph(n)
    seq_idx = torch.arange(n) % 21
    out = SAMPLER(edge_index, seq_idx)
    assert torch.all(out[:, :, 0] == WalkSampler.CLS_TOKEN)


def test_second_token_is_starting_residue_aa():
    """Position 1 in every walk must be the AA type of the source residue."""
    n = 8
    edge_index = chain_graph(n)
    seq_idx = torch.arange(n) % 21
    out = SAMPLER(edge_index, seq_idx)
    for i in range(n):
        assert torch.all(out[i, :, 1] == seq_idx[i])


def test_token_values_in_valid_range():
    """All token values must be in 0–21 (0–20 AA, 21 CLS)."""
    n = 15
    edge_index = chain_graph(n)
    seq_idx = torch.randint(0, 21, (n,))
    out = SAMPLER(edge_index, seq_idx)
    assert torch.all(out >= 0)
    assert torch.all(out <= WalkSampler.CLS_TOKEN)


def test_isolated_node_stays_put():
    """An isolated node (degree 0) must stay at itself for all walk steps."""
    # chain 0-1-2, node 3 is isolated
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    seq_idx = torch.tensor([5, 6, 7, 8])
    out = SAMPLER(edge_index, seq_idx)
    # node 3: position 0 = CLS, positions 1..6 = seq_idx[3] = 8
    assert torch.all(out[3, :, 0] == WalkSampler.CLS_TOKEN)
    assert torch.all(out[3, :, 1:] == 8)


def test_multiple_walks_can_differ():
    """On a branching graph, K > 1 walks from the same node should not all be equal."""
    # star graph: node 0 → {1,2,3,4} bidirectionally
    src = [0, 0, 0, 0, 1, 2, 3, 4]
    dst = [1, 2, 3, 4, 0, 0, 0, 0]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    seq_idx = torch.arange(5) % 21
    sampler = WalkSampler(num_walks=16, walk_length=3)
    out = sampler(edge_index, seq_idx)
    walks_from_hub = out[0]   # (16, 5)
    # Not all 16 walks should be identical (star has 4 choices at each step)
    assert not torch.all(walks_from_hub == walks_from_hub[0])
