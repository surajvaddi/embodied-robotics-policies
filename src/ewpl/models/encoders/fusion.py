"""Fusion encoder for multi-modal policy inputs."""

from __future__ import annotations

import torch
from torch import nn


class FusionEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat(embeddings, dim=-1))

