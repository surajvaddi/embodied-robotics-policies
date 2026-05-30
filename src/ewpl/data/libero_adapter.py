"""LIBERO dataset indexing and canonical conversion utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ewpl.data.schemas import Action, Episode, Observation, Step
from ewpl.sim.libero_env import LiberoEnv, instruction_for_task


DEFAULT_TASKS = [
    "put_the_bowl_on_the_plate",
    "put_the_mug_on_the_left_plate",
    "pick_up_the_black_bowl",
    "put_the_book_on_the_caddy",
]


def load_libero_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load the small Phase-2 LIBERO config without requiring PyYAML."""

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
    """Index LIBERO tasks and available demonstration files."""

    suite = str(config.get("suite", "libero_spatial"))
    root = Path(str(config.get("root", "data/raw/libero"))).expanduser()
    configured_tasks = list(config.get("tasks") or DEFAULT_TASKS)
    tasks = configured_tasks[:limit_tasks] if limit_tasks else configured_tasks

    demo_files = []
    if root.exists():
        patterns = ["*.hdf5", "*.h5", "*.npz", "*.json"]
        for pattern in patterns:
            demo_files.extend(str(path) for path in sorted(root.rglob(pattern)))

    manifest = {
        "source": "libero",
        "suite": suite,
        "root": str(root),
        "dry_run": dry_run,
        "tasks": [{"task_id": task_id, "instruction": instruction_for_task(task_id)} for task_id in tasks],
        "tasks_indexed": len(tasks),
        "demo_files_available": bool(demo_files),
        "demo_files": demo_files,
    }
    return manifest


def write_manifest(manifest: Dict[str, Any], out: Union[str, Path]) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def convert_episode(
    task_id: str,
    *,
    episode_idx: int,
    suite: str = "libero_spatial",
    steps: int = 16,
    seed: int = 0,
) -> Episode:
    """Convert one LIBERO rollout/demo into the canonical episode schema.

    Phase 2 uses the smoke environment to produce deterministic LIBERO-shaped
    episodes. The function boundary is intentionally the same place where real
    demonstration decoding will be attached once raw LIBERO files are available.
    """

    env = LiberoEnv(suite=suite)
    observation = env.reset(task_id=task_id, seed=seed)
    episode_steps: List[Step] = []
    rng = np.random.default_rng(seed)

    for t in range(steps):
        if t == steps - 1:
            action_vector = np.zeros(7, dtype=np.float32)
        else:
            action_vector = rng.normal(loc=0.0, scale=0.05, size=7).astype(np.float32)
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
                info={"suite": suite, "task_id": task_id, "smoke_conversion": True},
            )
        )
        if result is not None:
            observation = result.observation

    return Episode(
        episode_id=f"libero_{episode_idx:05d}",
        source="libero",
        task_id=task_id,
        instruction=instruction_for_task(task_id),
        steps=episode_steps,
        success=True,
        metadata={"suite": suite, "seed": seed, "conversion": "smoke"},
    )


def convert_to_canonical(
    config_path: Union[str, Path],
    out: Union[str, Path],
    *,
    limit_episodes: int = 20,
) -> List[Episode]:
    config = load_libero_config(config_path)
    manifest = index_tasks(config)
    tasks = [task["task_id"] for task in manifest["tasks"]]
    if not tasks:
        raise ValueError("LIBERO config did not resolve any tasks")

    episodes = []
    for idx in range(limit_episodes):
        task_id = tasks[idx % len(tasks)]
        episodes.append(
            convert_episode(
                task_id,
                episode_idx=idx,
                suite=str(config.get("suite", "libero_spatial")),
                steps=int(config.get("smoke_steps", 16)),
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

