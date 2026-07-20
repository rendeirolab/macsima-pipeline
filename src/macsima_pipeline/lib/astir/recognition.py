"""Recognition network q(cell type | expression) for the Astir model."""

from __future__ import annotations

import torch
from torch import nn


class TypeRecognitionNet(nn.Module):
    """Amortized variational posterior over the C+1 cell-type classes.

    A small MLP mapping a cell's (z-scored) expression to class probabilities:
    Linear(G -> hidden) -> leaky_relu -> Linear(hidden -> C+1) -> softmax.
    """

    def __init__(self, n_features: int, n_classes: int, hidden: int = 20) -> None:
        super().__init__()
        self.hidden = nn.Linear(n_features, hidden)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.nn.functional.leaky_relu(self.hidden(x))
        logits = self.out(h)
        return torch.softmax(logits, dim=-1), torch.log_softmax(logits, dim=-1)
