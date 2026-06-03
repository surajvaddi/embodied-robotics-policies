"""Baseline policies used by smoke rollouts and harness tests."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from ewpl.data.schemas import Action, Observation
from ewpl.models.policies.base import BasePolicy


class RandomPolicy(BasePolicy):
    """Gaussian random action policy sized from the current observation."""

    def __init__(
        self,
        *,
        seed: int = 0,
        scale: float = 0.1,
        convention: str = "delta_ee_pose_gripper",
    ) -> None:
        super().__init__()
        self.seed = seed
        self.scale = scale
        self.convention = convention
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        super().reset()
        self._rng = np.random.default_rng(self.seed)

    def act(self, observation: Observation) -> Action:
        self.state.step += 1
        vector = self._rng.normal(
            loc=0.0,
            scale=self.scale,
            size=observation.proprio.shape[0],
        ).astype(np.float32)
        return Action(vector=vector, convention=self.convention)

    def action_chunk(self, observation: Observation, horizon: int) -> np.ndarray:
        return np.stack([self.act(observation).vector for _ in range(horizon)])


class ReplayPolicy(BasePolicy):
    """Replay a fixed sequence of actions, repeating the final action if exhausted."""

    def __init__(
        self,
        actions: Iterable[np.ndarray],
        *,
        convention: str = "replay",
    ) -> None:
        super().__init__()
        self.actions: List[np.ndarray] = [np.asarray(action, dtype=np.float32) for action in actions]
        if not self.actions:
            raise ValueError("ReplayPolicy requires at least one action")
        self.convention = convention

    def act(self, observation: Observation) -> Action:
        idx = min(self.state.step, len(self.actions) - 1)
        self.state.step += 1
        return Action(vector=self.actions[idx].copy(), convention=self.convention)

    def action_chunk(self, observation: Observation, horizon: int) -> np.ndarray:
        return np.stack([self.act(observation).vector for _ in range(horizon)])


def make_policy(name: str, *, seed: int = 0, action_scale: float = 0.1):
    if name == "random":
        return RandomPolicy(seed=seed, scale=action_scale)
    raise ValueError(f"unsupported policy: {name}")

