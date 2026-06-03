"""Behavior cloning policy."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import nn

from ewpl.data.schemas import Action, Observation
from ewpl.data.tensorization import tensorize_observation
from ewpl.models.encoders.fusion import FusionEncoder
from ewpl.models.encoders.language import LanguageEncoder
from ewpl.models.encoders.proprio import ProprioEncoder
from ewpl.models.encoders.vision import VisionEncoder
from ewpl.models.policies.base import PolicyState


class BCPolicy(nn.Module):
    """Small supervised policy: RGB + language + proprio -> action vector."""

    def __init__(
        self,
        *,
        proprio_dim: int,
        action_dim: int,
        image_size: int = 32,
        language_length: int = 16,
        vocab_size: int = 4096,
        hidden_dim: int = 64,
        action_convention: str = "delta_ee_pose_gripper",
    ) -> None:
        super().__init__()
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.image_size = image_size
        self.language_length = language_length
        self.vocab_size = vocab_size
        self.action_convention = action_convention
        self.vision = VisionEncoder(output_dim=hidden_dim)
        self.language = LanguageEncoder(vocab_size=vocab_size, output_dim=hidden_dim)
        self.proprio = ProprioEncoder(input_dim=proprio_dim, output_dim=hidden_dim)
        self.fusion = FusionEncoder(input_dim=hidden_dim * 3, output_dim=hidden_dim * 2)
        self.head = nn.Linear(hidden_dim * 2, action_dim)
        self._state = PolicyState()

    @property
    def state(self) -> PolicyState:
        return self._state

    def reset(self) -> None:
        self._state = PolicyState()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        image_emb = self.vision(batch["image"].float())
        lang_emb = self.language(batch["language_ids"].long())
        prop_emb = self.proprio(batch["proprio"].float())
        fused = self.fusion(image_emb, lang_emb, prop_emb)
        return self.head(fused)

    @torch.no_grad()
    def act(self, observation: Observation) -> Action:
        self.eval()
        sample = tensorize_observation(
            observation,
            image_size=self.image_size,
            language_length=self.language_length,
            vocab_size=self.vocab_size,
        )
        batch = {
            "image": torch.from_numpy(sample["image"]).unsqueeze(0),
            "language_ids": torch.from_numpy(sample["language_ids"]).unsqueeze(0),
            "proprio": torch.from_numpy(sample["proprio"]).unsqueeze(0),
        }
        action = self.forward(batch).squeeze(0).cpu().numpy().astype(np.float32)
        self._state.step += 1
        return Action(vector=action, convention=self.action_convention)

    @torch.no_grad()
    def action_chunk(self, observation: Observation, horizon: int) -> np.ndarray:
        return np.stack([self.act(observation).vector for _ in range(horizon)])

