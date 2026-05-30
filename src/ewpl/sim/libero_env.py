"""LIBERO environment adapter.

The adapter is structured so a real LIBERO backend can be plugged in later.
For local tests and smoke commands, it provides a deterministic lightweight
backend with the same canonical interface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ewpl.data.schemas import Action, Observation
from ewpl.sim.env_base import StepResult


class LiberoEnv:
    """Canonical wrapper around a LIBERO-like manipulation environment."""

    def __init__(self, suite: str = "libero_spatial", backend: Optional[Any] = None) -> None:
        self.suite = suite
        self.backend = backend
        self.task_id = ""
        self.seed = 0
        self.t = 0
        self._success = False
        self._rng = np.random.default_rng(0)
        self._last_frame = np.zeros((64, 64, 3), dtype=np.uint8)

    def reset(self, task_id: str, seed: int = 0) -> Observation:
        self.task_id = task_id
        self.seed = seed
        self.t = 0
        self._success = False
        self._rng = np.random.default_rng(seed)

        if self.backend is not None:
            raw_obs = self.backend.reset(task_id=task_id, seed=seed)
            return observation_from_libero(raw_obs, task_id=task_id)

        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        if self.backend is not None:
            raw = self.backend.step(action.vector)
            observation = observation_from_libero(raw["observation"], task_id=self.task_id)
            return StepResult(
                observation=observation,
                reward=float(raw.get("reward", 0.0)),
                done=bool(raw.get("done", False)),
                success=bool(raw.get("success", False)),
                info=dict(raw.get("info", {})),
            )

        self.t += 1
        self._success = self.t >= 3 and float(np.linalg.norm(action.vector)) < 10.0
        done = self.t >= 4 or self._success
        observation = self._make_observation()
        return StepResult(
            observation=observation,
            reward=1.0 if self._success else 0.0,
            done=done,
            success=self._success,
            info={"suite": self.suite, "task_id": self.task_id, "t": self.t},
        )

    def render(self, camera: str = "agentview") -> np.ndarray:
        if self.backend is not None:
            return np.asarray(self.backend.render(camera=camera), dtype=np.uint8)
        return self._last_frame.copy()

    def is_success(self) -> bool:
        if self.backend is not None and hasattr(self.backend, "is_success"):
            return bool(self.backend.is_success())
        return self._success

    def get_task_metadata(self) -> Dict[str, Any]:
        return {
            "source": "libero",
            "suite": self.suite,
            "task_id": self.task_id,
            "seed": self.seed,
            "backend": "external" if self.backend is not None else "smoke",
        }

    def _make_observation(self) -> Observation:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :] = np.array([36, 42, 52], dtype=np.uint8)
        frame[44:54, 8:56] = np.array([80, 90, 100], dtype=np.uint8)

        block_x = min(52, 8 + self.t * 9)
        frame[24:34, block_x : block_x + 10] = np.array([220, 60, 40], dtype=np.uint8)
        frame[22:38, 46:58] = np.array([40, 90, 220], dtype=np.uint8)

        self._last_frame = frame
        proprio = self._rng.normal(loc=0.0, scale=0.01, size=7).astype(np.float32)
        proprio[0] = float(self.t)
        return Observation(
            rgb=frame,
            depth=None,
            proprio=proprio,
            language=instruction_for_task(self.task_id),
            camera_intrinsics={"fx": 64.0, "fy": 64.0, "cx": 32.0, "cy": 32.0},
            camera_extrinsics={"camera": "agentview"},
            sim_state={"suite": self.suite, "task_id": self.task_id, "t": self.t},
        )


def instruction_for_task(task_id: str) -> str:
    normalized = task_id.replace("_", " ").strip()
    return f"complete LIBERO task: {normalized or 'unknown task'}"


def observation_from_libero(raw: Dict[str, Any], task_id: str) -> Observation:
    """Map a LIBERO-style observation dictionary into the canonical schema."""

    rgb = raw.get("rgb")
    if rgb is None:
        rgb = raw.get("agentview_image")
    if rgb is None:
        rgb = raw.get("image")
    if rgb is None:
        raise ValueError("LIBERO observation is missing an RGB image")

    proprio = raw.get("proprio")
    if proprio is None:
        proprio = raw.get("robot_state")
    if proprio is None:
        proprio = raw.get("state")
    if proprio is None:
        proprio = np.zeros(7, dtype=np.float32)

    return Observation(
        rgb=np.asarray(rgb, dtype=np.uint8),
        depth=raw.get("depth"),
        proprio=np.asarray(proprio, dtype=np.float32),
        language=str(raw.get("language") or raw.get("instruction") or instruction_for_task(task_id)),
        camera_intrinsics=raw.get("camera_intrinsics"),
        camera_extrinsics=raw.get("camera_extrinsics"),
        sim_state=raw.get("sim_state"),
    )
