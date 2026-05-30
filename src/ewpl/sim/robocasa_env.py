"""RoboCasa environment adapter with a deterministic smoke backend."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ewpl.data.schemas import Action, Observation
from ewpl.sim.env_base import StepResult


class RoboCasaEnv:
    """Canonical wrapper around a RoboCasa-like household manipulation task."""

    def __init__(self, benchmark: str = "robocasa365", backend: Optional[Any] = None) -> None:
        self.benchmark = benchmark
        self.backend = backend
        self.task_id = ""
        self.scene_id = ""
        self.seed = 0
        self.t = 0
        self._success = False
        self._rng = np.random.default_rng(0)
        self._last_frame = np.zeros((80, 80, 3), dtype=np.uint8)

    def reset(self, task_id: str, scene_id: str = "kitchen_scene_000", seed: int = 0) -> Observation:
        self.task_id = task_id
        self.scene_id = scene_id
        self.seed = seed
        self.t = 0
        self._success = False
        self._rng = np.random.default_rng(seed)

        if self.backend is not None:
            raw_obs = self.backend.reset(task_id=task_id, scene_id=scene_id, seed=seed)
            return observation_from_robocasa(raw_obs, task_id=task_id, scene_id=scene_id)

        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        if self.backend is not None:
            raw = self.backend.step(action.vector)
            observation = observation_from_robocasa(
                raw["observation"], task_id=self.task_id, scene_id=self.scene_id
            )
            return StepResult(
                observation=observation,
                reward=float(raw.get("reward", 0.0)),
                done=bool(raw.get("done", False)),
                success=bool(raw.get("success", False)),
                info=dict(raw.get("info", {})),
            )

        self.t += 1
        self._success = self.t >= 4 and float(np.linalg.norm(action.vector)) < 10.0
        done = self.t >= 6 or self._success
        observation = self._make_observation()
        return StepResult(
            observation=observation,
            reward=1.0 if self._success else 0.0,
            done=done,
            success=self._success,
            info={
                "benchmark": self.benchmark,
                "task_id": self.task_id,
                "scene_id": self.scene_id,
                "t": self.t,
            },
        )

    def render(self, camera: str = "robot0_agentview_left") -> np.ndarray:
        if self.backend is not None:
            return np.asarray(self.backend.render(camera=camera), dtype=np.uint8)
        return self._last_frame.copy()

    def is_success(self) -> bool:
        if self.backend is not None and hasattr(self.backend, "is_success"):
            return bool(self.backend.is_success())
        return self._success

    def get_task_metadata(self) -> Dict[str, Any]:
        return {
            "source": "robocasa",
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "scene_id": self.scene_id,
            "seed": self.seed,
            "backend": "external" if self.backend is not None else "smoke",
        }

    def _make_observation(self) -> Observation:
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        frame[:, :] = np.array([58, 54, 48], dtype=np.uint8)
        frame[50:66, 8:72] = np.array([105, 96, 82], dtype=np.uint8)
        frame[12:50, 6:16] = np.array([120, 126, 118], dtype=np.uint8)
        frame[12:50, 64:74] = np.array([120, 126, 118], dtype=np.uint8)

        object_x = min(62, 10 + self.t * 8)
        frame[34:44, object_x : object_x + 10] = np.array([220, 170, 40], dtype=np.uint8)
        frame[28:46, 54:70] = np.array([55, 120, 190], dtype=np.uint8)

        self._last_frame = frame
        proprio = self._rng.normal(loc=0.0, scale=0.02, size=9).astype(np.float32)
        proprio[0] = float(self.t)
        return Observation(
            rgb=frame,
            depth=None,
            proprio=proprio,
            language=instruction_for_task(self.task_id),
            camera_intrinsics={"fx": 80.0, "fy": 80.0, "cx": 40.0, "cy": 40.0},
            camera_extrinsics={"camera": "robot0_agentview_left"},
            sim_state={
                "benchmark": self.benchmark,
                "task_id": self.task_id,
                "scene_id": self.scene_id,
                "t": self.t,
            },
        )


def instruction_for_task(task_id: str) -> str:
    normalized = task_id.replace("_", " ").strip()
    return f"complete RoboCasa task: {normalized or 'unknown task'}"


def observation_from_robocasa(raw: Dict[str, Any], task_id: str, scene_id: str) -> Observation:
    """Map a RoboCasa-style observation dictionary into the canonical schema."""

    rgb = raw.get("rgb")
    if rgb is None:
        rgb = raw.get("agentview_image")
    if rgb is None:
        rgb = raw.get("robot0_agentview_left_image")
    if rgb is None:
        rgb = raw.get("image")
    if rgb is None:
        raise ValueError("RoboCasa observation is missing an RGB image")

    proprio = raw.get("proprio")
    if proprio is None:
        proprio = raw.get("robot_state")
    if proprio is None:
        proprio = raw.get("state")
    if proprio is None:
        proprio = np.zeros(9, dtype=np.float32)

    sim_state = dict(raw.get("sim_state") or {})
    sim_state.setdefault("scene_id", scene_id)

    return Observation(
        rgb=np.asarray(rgb, dtype=np.uint8),
        depth=raw.get("depth"),
        proprio=np.asarray(proprio, dtype=np.float32),
        language=str(raw.get("language") or raw.get("instruction") or instruction_for_task(task_id)),
        camera_intrinsics=raw.get("camera_intrinsics"),
        camera_extrinsics=raw.get("camera_extrinsics"),
        sim_state=sim_state,
    )

