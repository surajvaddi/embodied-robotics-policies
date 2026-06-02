"""LeRobot / Hugging Face Hub dataset access.

This module keeps LeRobot as an optional dependency. When it is installed, the
factory can build streaming or local LeRobot datasets. When it is not installed,
the rest of the project can still import this module and run smoke tests.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class LeRobotConfig:
    """Small config object for online-first LeRobot ingestion."""

    source: str
    backend: str
    streaming: bool
    seed: int
    default_repo_id: str
    robocasa_repo_id: str
    max_streamed_steps_per_episode: int
    action_convention: str
    task_id: str
    instruction: str
    image_keys: List[str]
    state_keys: List[str]
    action_keys: List[str]


@dataclass
class LeRobotAvailability:
    """Import status for optional LeRobot dataset classes."""

    lerobot_installed: bool
    streaming_available: bool
    local_dataset_available: bool
    detail: str


def load_lerobot_config(path: Union[str, Path]) -> LeRobotConfig:
    payload = _load_simple_yaml(path)
    return LeRobotConfig(
        source=str(payload.get("source", "lerobot")),
        backend=str(payload.get("backend", "huggingface")),
        streaming=bool(payload.get("streaming", True)),
        seed=int(payload.get("seed", 0)),
        default_repo_id=str(payload.get("default_repo_id", "lerobot/pusht")),
        robocasa_repo_id=str(payload.get("robocasa_repo_id", "pepijn223/robocasa_CloseFridge")),
        max_streamed_steps_per_episode=int(payload.get("max_streamed_steps_per_episode", 32)),
        action_convention=str(payload.get("action_convention", "lerobot_continuous")),
        task_id=str(payload.get("task_id", "online_lerobot_sample")),
        instruction=str(payload.get("instruction", "online LeRobot demonstration")),
        image_keys=list(payload.get("image_keys") or []),
        state_keys=list(payload.get("state_keys") or []),
        action_keys=list(payload.get("action_keys") or []),
    )


def check_lerobot_availability() -> LeRobotAvailability:
    streaming_available = _module_has_class(
        "lerobot.datasets.streaming_dataset", "StreamingLeRobotDataset"
    )
    local_available = _module_has_class("lerobot.datasets.lerobot_dataset", "LeRobotDataset")
    installed = streaming_available or local_available or importlib.util.find_spec("lerobot") is not None
    detail = "installed" if installed else "lerobot is not installed"
    return LeRobotAvailability(
        lerobot_installed=installed,
        streaming_available=streaming_available,
        local_dataset_available=local_available,
        detail=detail,
    )


def make_lerobot_dataset(
    repo_id: str,
    *,
    streaming: bool = True,
    root: Optional[Union[str, Path]] = None,
) -> Any:
    """Create a LeRobot dataset object using optional runtime imports."""

    if streaming:
        module = importlib.import_module("lerobot.datasets.streaming_dataset")
        dataset_cls = getattr(module, "StreamingLeRobotDataset")
        return dataset_cls(repo_id)

    module = importlib.import_module("lerobot.datasets.lerobot_dataset")
    dataset_cls = getattr(module, "LeRobotDataset")
    if root is None:
        return dataset_cls(repo_id)
    return dataset_cls(repo_id, root=root)


def configured_repo_id(config: LeRobotConfig, *, source_family: str = "lerobot") -> str:
    if source_family == "robocasa":
        return config.robocasa_repo_id
    return config.default_repo_id


def first_present(sample: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in sample:
            return sample[key]
    return None


def _module_has_class(module_name: str, class_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    return hasattr(module, class_name)


def _load_simple_yaml(path: Union[str, Path]) -> Dict[str, Any]:
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


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    return value.strip("\"'")

