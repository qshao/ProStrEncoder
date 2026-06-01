import torch.nn as nn


class ResidueHead(nn.Module):
    """
    Linear head for masked residue type prediction.
    Takes per-residue embeddings and projects to 21-class logits.
    """

    def __init__(self, in_dim: int, num_classes: int = 21):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, embeddings):
        return self.linear(embeddings)
