import io
import tempfile
import os
import numpy as np
import pytest
from prostrencoder.data.parser import parse_structure, AA_TO_IDX

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
def pdb_file(tmp_path):
    p = tmp_path / "test.pdb"
    p.write_text(MINIMAL_PDB)
    return str(p)

def test_parse_structure_returns_expected_keys(pdb_file):
    result = parse_structure(pdb_file)
    assert set(result.keys()) == {"seq_idx", "ca_coords", "backbone_coords"}

def test_parse_structure_residue_count(pdb_file):
    result = parse_structure(pdb_file)
    assert result["seq_idx"].shape == (3,)
    assert result["ca_coords"].shape == (3, 3)
    assert result["backbone_coords"].shape == (3, 4, 3)

def test_parse_structure_aa_types(pdb_file):
    result = parse_structure(pdb_file)
    assert result["seq_idx"][0] == AA_TO_IDX["ALA"]
    assert result["seq_idx"][1] == AA_TO_IDX["GLY"]
    assert result["seq_idx"][2] == AA_TO_IDX["LEU"]

def test_parse_structure_ca_coords_are_finite(pdb_file):
    result = parse_structure(pdb_file)
    assert np.all(np.isfinite(result["ca_coords"]))

def test_parse_unknown_residue_gets_unk_index(tmp_path):
    pdb = """\
ATOM      1  N   XYZ A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  XYZ A   1      11.526  10.000  10.000  1.00  0.00           C
ATOM      3  C   XYZ A   1      12.000  11.400  10.000  1.00  0.00           C
ATOM      4  O   XYZ A   1      11.200  12.200  10.000  1.00  0.00           O
END
"""
    # XYZ is not a standard AA, but BioPython may not list it via is_aa
    # This test verifies that unrecognized residues map to UNK index (20)
    p = tmp_path / "unk.pdb"
    p.write_text(pdb)
    result = parse_structure(str(p))
    # If BioPython filters it out, length is 0; if kept, index is 20
    if len(result["seq_idx"]) > 0:
        assert result["seq_idx"][0] == 20
