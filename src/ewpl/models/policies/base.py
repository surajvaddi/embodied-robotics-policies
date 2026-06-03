"""Policy interface used by rollout and training code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

import numpy as np

from ewpl.data.schemas import Action, Observation


@dataclass
class PolicyState:
    """Lightweight per-episode state exposed for logging/debugging."""

    step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Policy(Protocol):
    """Common policy API for random, scripted, replay, and learned policies."""

    def reset(self) -> None:
        """Clear any per-episode policy state."""

    def act(self, observation: Observation) -> Action:
        """Return the next action for one canonical observation."""

    def action_chunk(self, observation: Observation, horizon: int) -> Optional[np.ndarray]:
        """Optionally return a future action chunk with shape [horizon, action_dim]."""

    @property
    def state(self) -> PolicyState:
        """Return inspectable policy state for logs."""


class BasePolicy:
    """Convenience base class for simple non-neural policies."""

    def __init__(self) -> None:
        self._state = PolicyState()

    @property
    def state(self) -> PolicyState:
        return self._state

    def reset(self) -> None:
        self._state = PolicyState()

    def action_chunk(self, observation: Observation, horizon: int) -> Optional[np.ndarray]:
        return None

