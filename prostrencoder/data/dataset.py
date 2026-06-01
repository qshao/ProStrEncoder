import os
import torch
from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    """
    Loads preprocessed PyG Data objects from a directory of `.pt` files.
    Each file was created by scripts/preprocess.py.
    """

    def __init__(self, processed_dir: str):
        self.files = sorted([
            os.path.join(processed_dir, f)
            for f in os.listdir(processed_dir)
            if f.endswith(".pt")
        ])
        if not self.files:
            raise RuntimeError(f"No .pt files found in {processed_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=False)
