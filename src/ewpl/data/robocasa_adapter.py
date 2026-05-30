"""RoboCasa dataset indexing and canonical conversion utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

