"""Proprioceptive state encoder."""

from __future__ import annotations

import torch
from torch import nn


class ProprioEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio)

