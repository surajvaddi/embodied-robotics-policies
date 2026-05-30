"""Base interfaces for robotics simulation adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

import numpy as np

from ewpl.data.schemas import Action, Observation


@dataclass
class StepResult:
    """Result returned by environment adapters after one action."""

    observation: Observation
    reward: float
    done: bool
    success: bool
    info: Dict[str, Any]


class EnvAdapter(Protocol):
    """Common interface for LIBERO, RoboCasa, and future simulator wrappers."""

    def reset(self, task_id: str, seed: int = 0) -> Observation:
        """Reset a task and return the first canonical observation."""

    def step(self, action: Action) -> StepResult:
        """Advance the environment by one canonical action."""

    def render(self, camera: str = "agentview") -> np.ndarray:
        """Render an RGB frame for a named camera."""

    def is_success(self) -> bool:
        """Return whether the current rollout has succeeded."""

    def get_task_metadata(self) -> Dict[str, Any]:
        """Return task metadata useful for conversion/evaluation."""

