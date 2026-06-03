"""Small language encoder based on hashed token ids."""

from __future__ import annotations

import torch
from torch import nn


class LanguageEncoder(nn.Module):
    def __init__(self, vocab_size: int = 4096, embed_dim: int = 32, output_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Sequential(nn.Linear(embed_dim, output_dim), nn.ReLU())

    def forward(self, language_ids: torch.Tensor) -> torch.Tensor:
        mask = (language_ids != 0).float().unsqueeze(-1)
        embedded = self.embedding(language_ids)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (embedded * mask).sum(dim=1) / denom
        return self.proj(pooled)

