"""RoboCasa dataset indexing and canonical conversion utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ewpl.data.schemas import Action, Episode, Step
from ewpl.sim.robocasa_env import RoboCasaEnv
from ewpl.sim.robocasa_env import instruction_for_task


DEFAULT_TASKS = [
    "open_the_top_drawer",
    "close_the_microwave",
    "place_the_mug_in_the_sink",
    "turn_on_the_stove",
]

DEFAULT_SCENES = [
    "kitchen_scene_000",
    "kitchen_scene_001",
    "kitchen_scene_002",
]


def load_robocasa_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load the small Phase-3 RoboCasa config without requiring PyYAML."""

    config_path = Path(path)
    payload: Dict[str, Any] = {}
    current_list_key: Optional[str] = None
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].rstrip()
        if not line:
            continue
        if line.startswith("  - ") and current_list_key:
            payload.setdefault(current_list_key, []).append(_parse_scalar(line[4:].strip()))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if value == "":
            payload[key] = []
            current_list_key = key
        else:
            payload[key] = _parse_scalar(value)
            current_list_key = None
    return payload


def index_tasks(
    config: Dict[str, Any], *, limit_tasks: Optional[int] = None, dry_run: bool = False
) -> Dict[str, Any]:
    """Index RoboCasa tasks, scene variations, and available demonstration files."""

    benchmark = str(config.get("benchmark", "robocasa365"))
    root = Path(str(config.get("root", "data/raw/robocasa"))).expanduser()
    configured_tasks = list(config.get("tasks") or DEFAULT_TASKS)
    configured_scenes = list(config.get("scenes") or DEFAULT_SCENES)
    tasks = configured_tasks[:limit_tasks] if limit_tasks else configured_tasks

    demo_files: List[str] = []
    if root.exists():
        patterns = ["*.hdf5", "*.h5", "*.npz", "*.json"]
        for pattern in patterns:
            demo_files.extend(str(path) for path in sorted(root.rglob(pattern)))

    return {
        "source": "robocasa",
        "benchmark": benchmark,
        "root": str(root),
        "dry_run": dry_run,
        "tasks": [{"task_id": task_id, "instruction": instruction_for_task(task_id)} for task_id in tasks],
        "tasks_indexed": len(tasks),
        "scenes": configured_scenes,
        "scene_variations_indexed": len(configured_scenes),
        "demo_files_available": bool(demo_files),
        "demo_files": demo_files,
    }


def write_manifest(manifest: Dict[str, Any], out: Union[str, Path]) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def convert_episode(
    task_id: str,
    scene_id: str,
    *,
    episode_idx: int,
    benchmark: str = "robocasa365",
    steps: int = 18,
    seed: int = 0,
) -> Episode:
    """Convert one RoboCasa rollout/demo into the canonical episode schema."""

    env = RoboCasaEnv(benchmark=benchmark)
    observation = env.reset(task_id=task_id, scene_id=scene_id, seed=seed)
    episode_steps: List[Step] = []
    rng = np.random.default_rng(seed)

    for t in range(steps):
        if t == steps - 1:
            action_vector = np.zeros(9, dtype=np.float32)
        else:
            action_vector = rng.normal(loc=0.0, scale=0.05, size=9).astype(np.float32)
        action = Action(vector=action_vector, convention="delta_ee_pose_gripper")
        result = env.step(action) if t < steps - 1 else None
        done = t == steps - 1
        episode_steps.append(
            Step(
                t=t,
                observation=observation,
                action=action,
                reward=1.0 if done else 0.0,
                done=done,
                info={
                    "benchmark": benchmark,
                    "task_id": task_id,
                    "scene_id": scene_id,
                    "smoke_conversion": True,
                },
            )
        )
        if result is not None:
            observation = result.observation

    return Episode(
        episode_id=f"robocasa_{episode_idx:05d}",
        source="robocasa",
        task_id=task_id,
        instruction=instruction_for_task(task_id),
        steps=episode_steps,
        success=True,
        metadata={
            "benchmark": benchmark,
            "scene_id": scene_id,
            "seed": seed,
            "conversion": "smoke",
        },
    )


def convert_to_canonical(
    config_path: Union[str, Path],
    out: Union[str, Path],
    *,
    limit_episodes: int = 20,
) -> List[Episode]:
    config = load_robocasa_config(config_path)
    manifest = index_tasks(config)
    tasks = [task["task_id"] for task in manifest["tasks"]]
    scenes = list(manifest["scenes"])
    if not tasks:
        raise ValueError("RoboCasa config did not resolve any tasks")
    if not scenes:
        raise ValueError("RoboCasa config did not resolve any scenes")

    episodes = []
    for idx in range(limit_episodes):
        task_id = tasks[idx % len(tasks)]
        scene_id = scenes[idx % len(scenes)]
        episodes.append(
            convert_episode(
                task_id,
                scene_id,
                episode_idx=idx,
                benchmark=str(config.get("benchmark", "robocasa365")),
                steps=int(config.get("smoke_steps", 18)),
                seed=int(config.get("seed", 0)) + idx,
            )
        )
    return episodes


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")
