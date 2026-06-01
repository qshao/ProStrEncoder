import os
import numpy as np
import torch
import pytest
from torch_geometric.data import Data
from prostrencoder.data.dataset import ProteinDataset

MINIMAL_PDB = """\
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.526  10.000  10.000  1.00  0.00           C
ATOM      3  C   ALA A   1      12.000  11.400  10.000  1.00  0.00           C
ATOM      4  O   ALA A   1      11.200  12.200  10.000  1.00  0.00           O
ATOM      5  N   GLY A   2      13.300  11.700  10.000  1.00  0.00           N
ATOM      6  CA  GLY A   2      13.800  13.100  10.000  1.00  0.00           C
ATOM      7  C   GLY A   2      15.300  13.100  10.000  1.00  0.00           C
ATOM      8  O   GLY A   2      15.900  12.000  10.000  1.00  0.00           O
ATOM      9  N   LEU A   3      15.900  14.300  10.000  1.00  0.00           N
ATOM     10  CA  LEU A   3      17.400  14.300  10.000  1.00  0.00           C
ATOM     11  C   LEU A   3      17.900  15.700  10.000  1.00  0.00           C
ATOM     12  O   LEU A   3      17.100  16.600  10.000  1.00  0.00           O
END
"""


@pytest.fixture
def processed_dir(tmp_path):
    """Build one preprocessed .pt file matching the new preprocess_one() output."""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text(MINIMAL_PDB)

    from prostrencoder.data.parser import parse_structure
    from prostrencoder.data.graph_builder import build_knn_graph
    from prostrencoder.data.features import (
        aa_one_hot, compute_backbone_frame, compute_torsion_angles,
        rbf_encoding, compute_edge_directions,
    )

    parsed = parse_structure(str(pdb_path))
    edge_index, edge_dist = build_knn_graph(parsed["ca_coords"], k=30)
    onehot  = aa_one_hot(parsed["seq_idx"])
    torsion = compute_torsion_angles(parsed["backbone_coords"])
    frame   = compute_backbone_frame(parsed["backbone_coords"])
    rbf     = rbf_encoding(edge_dist)
    dirs    = compute_edge_directions(parsed["ca_coords"], edge_index, edge_dist)

    x_scalar = np.concatenate([onehot, torsion], axis=1).astype(np.float32)  # (N, 27)
    e_vec    = dirs[:, np.newaxis, :].astype(np.float32)                      # (E, 1, 3)

    data = Data(
        seq_idx=torch.from_numpy(parsed["seq_idx"]),
        x_scalar=torch.from_numpy(x_scalar),
        x_vec=torch.from_numpy(frame),
        edge_index=torch.from_numpy(edge_index),
        edge_scalar=torch.from_numpy(rbf),
        edge_vec=torch.from_numpy(e_vec),
    )
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()
    torch.save(data, proc_dir / "test.pt")
    return str(proc_dir)


def test_dataset_length(processed_dir):
    ds = ProteinDataset(processed_dir)
    assert len(ds) == 1


def test_dataset_item_keys(processed_dir):
    ds = ProteinDataset(processed_dir)
    item = ds[0]
    assert hasattr(item, "seq_idx")
    assert hasattr(item, "x_scalar")
    assert hasattr(item, "x_vec")
    assert hasattr(item, "edge_index")
    assert hasattr(item, "edge_scalar")
    assert hasattr(item, "edge_vec")


def test_dataset_feature_dims(processed_dir):
    ds = ProteinDataset(processed_dir)
    item = ds[0]
    N = item.seq_idx.shape[0]
    assert item.x_scalar.shape == (N, 27)   # 21 AA + 6 torsion (no RWSE)
    assert item.x_vec.shape == (N, 3, 3)
    assert item.edge_scalar.shape[1] == 16
    assert item.edge_vec.shape[1:] == (1, 3)
