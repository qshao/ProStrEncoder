import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa

AA_CODES = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL", "UNK",
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_CODES)}


def parse_structure(filepath: str) -> dict:
    """
    Parse a PDB or mmCIF file and return per-residue arrays for all standard
    amino acid residues found in the first model, across all chains.

    Returns a dict with:
        seq_idx         : np.ndarray (N,) int64   — AA index (20 = UNK)
        ca_coords       : np.ndarray (N, 3) float32 — Cα coordinates (Å)
        backbone_coords : np.ndarray (N, 4, 3) float32 — N, CA, C, O coords
    """
    if filepath.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("prot", filepath)
    try:
        model = next(structure.get_models())
    except StopIteration:
        raise ValueError(f"No models found in {filepath}")

    seq_idx_list, ca_list, bb_list = [], [], []

    for chain in model:
        for res in chain:
            if not is_aa(res, standard=True):
                continue

            resname = res.get_resname().strip()
            seq_idx_list.append(AA_TO_IDX.get(resname, 20))

            if "CA" in res:
                ca_list.append(res["CA"].get_vector().get_array())
            else:
                ca_list.append(np.full(3, np.nan, dtype=np.float32))

            bb = []
            for atom_name in ("N", "CA", "C", "O"):
                if atom_name in res:
                    bb.append(res[atom_name].get_vector().get_array())
                else:
                    bb.append(np.full(3, np.nan, dtype=np.float32))
            bb_list.append(bb)

    return {
        "seq_idx": np.array(seq_idx_list, dtype=np.int64),
        "ca_coords": np.array(ca_list, dtype=np.float32),
        "backbone_coords": np.array(bb_list, dtype=np.float32),
    }
