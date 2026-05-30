"""Canonical schema definitions for robot-learning episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np

EpisodeSource = Literal["libero", "robocasa", "openx", "lerobot", "synthetic"]


def _array(value: Any, *, name: str, ndim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(value)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {arr.ndim}")
    return arr


def _metadata(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


@dataclass
class Observation:
    """One canonical observation at a timestep."""

    rgb: Union[np.ndarray, str]
    depth: Optional[Union[np.ndarray, str]]
    proprio: np.ndarray
    language: str
    camera_intrinsics: Optional[Dict[str, Any]] = None
    camera_extrinsics: Optional[Dict[str, Any]] = None
    sim_state: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if isinstance(self.rgb, str):
            if not self.rgb:
                raise ValueError("rgb path cannot be empty")
        else:
            rgb = _array(self.rgb, name="rgb")
            if rgb.ndim not in {3, 4}:
                raise ValueError("rgb must be HWC, CHW, or multi-camera/video tensor")
            self.rgb = rgb

        if self.depth is not None and not isinstance(self.depth, str):
            self.depth = _array(self.depth, name="depth")

        self.proprio = _array(self.proprio, name="proprio", ndim=1).astype(np.float32)
        if not isinstance(self.language, str) or not self.language:
            raise ValueError("language must be a non-empty string")
        self.camera_intrinsics = _metadata(self.camera_intrinsics)
        self.camera_extrinsics = _metadata(self.camera_extrinsics)
        self.sim_state = _metadata(self.sim_state)


@dataclass
class Action:
    """Canonical robot action."""

    vector: np.ndarray
    convention: str
    gripper: Optional[Union[float, int]] = None

    def __post_init__(self) -> None:
        self.vector = _array(self.vector, name="action.vector", ndim=1).astype(np.float32)
        if not isinstance(self.convention, str) or not self.convention:
            raise ValueError("action convention must be a non-empty string")


@dataclass
class Step:
    """One transition in a canonical episode."""

    t: int
    observation: Observation
    action: Action
    reward: Optional[float]
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.t < 0:
            raise ValueError("step index must be non-negative")
        if self.reward is not None:
            self.reward = float(self.reward)
        self.done = bool(self.done)
        self.info = _metadata(self.info)


@dataclass
class Episode:
    """Canonical robotics episode used across simulation and offline datasets."""

    episode_id: str
    source: EpisodeSource
    task_id: str
    instruction: str
    steps: List[Step]
    success: Optional[bool]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id cannot be empty")
        if not self.task_id:
            raise ValueError("task_id cannot be empty")
        if not self.instruction:
            raise ValueError("instruction cannot be empty")
        if not self.steps:
            raise ValueError("episode must contain at least one step")
        expected = list(range(len(self.steps)))
        actual = [step.t for step in self.steps]
        if actual != expected:
            raise ValueError(f"step indices must be contiguous from 0, got {actual}")
        if self.steps[-1].done is not True:
            raise ValueError("final step must be marked done")
        self.success = None if self.success is None else bool(self.success)
        self.metadata = _metadata(self.metadata)
