"""Serialization helpers for canonical episodes."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ewpl.data.schemas import Action, Episode, Observation, Step


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def observation_to_dict(observation: Observation) -> Dict[str, Any]:
    return {
        "rgb": _to_jsonable(observation.rgb),
        "depth": _to_jsonable(observation.depth),
        "proprio": _to_jsonable(observation.proprio),
        "language": observation.language,
        "camera_intrinsics": _to_jsonable(observation.camera_intrinsics),
        "camera_extrinsics": _to_jsonable(observation.camera_extrinsics),
        "sim_state": _to_jsonable(observation.sim_state),
    }


def observation_from_dict(payload: Dict[str, Any]) -> Observation:
    return Observation(
        rgb=payload["rgb"],
        depth=payload.get("depth"),
        proprio=np.asarray(payload["proprio"], dtype=np.float32),
        language=payload["language"],
        camera_intrinsics=payload.get("camera_intrinsics"),
        camera_extrinsics=payload.get("camera_extrinsics"),
        sim_state=payload.get("sim_state"),
    )


def action_to_dict(action: Action) -> Dict[str, Any]:
    return {
        "vector": _to_jsonable(action.vector),
        "convention": action.convention,
        "gripper": action.gripper,
    }


def action_from_dict(payload: Dict[str, Any]) -> Action:
    return Action(
        vector=np.asarray(payload["vector"], dtype=np.float32),
        convention=payload["convention"],
        gripper=payload.get("gripper"),
    )


def step_to_dict(step: Step) -> Dict[str, Any]:
    return {
        "t": step.t,
        "observation": observation_to_dict(step.observation),
        "action": action_to_dict(step.action),
        "reward": step.reward,
        "done": step.done,
        "info": _to_jsonable(step.info),
    }


def step_from_dict(payload: Dict[str, Any]) -> Step:
    return Step(
        t=int(payload["t"]),
        observation=observation_from_dict(payload["observation"]),
        action=action_from_dict(payload["action"]),
        reward=payload.get("reward"),
        done=bool(payload["done"]),
        info=payload.get("info") or {},
    )


def episode_to_dict(episode: Episode) -> Dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "source": episode.source,
        "task_id": episode.task_id,
        "instruction": episode.instruction,
        "steps": [step_to_dict(step) for step in episode.steps],
        "success": episode.success,
        "metadata": _to_jsonable(episode.metadata),
    }


def episode_from_dict(payload: Dict[str, Any]) -> Episode:
    return Episode(
        episode_id=payload["episode_id"],
        source=payload["source"],
        task_id=payload["task_id"],
        instruction=payload["instruction"],
        steps=[step_from_dict(step) for step in payload["steps"]],
        success=payload.get("success"),
        metadata=payload.get("metadata") or {},
    )


__all__ = [
    "Action",
    "Episode",
    "Observation",
    "Step",
    "episode_from_dict",
    "episode_to_dict",
]
