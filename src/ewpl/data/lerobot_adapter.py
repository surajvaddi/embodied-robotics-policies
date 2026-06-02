"""LeRobot / Hugging Face Hub dataset access.

This module keeps LeRobot as an optional dependency. When it is installed, the
factory can build streaming or local LeRobot datasets. When it is not installed,
the rest of the project can still import this module and run smoke tests.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ewpl.data.schemas import Action, Episode, Observation, Step


@dataclass
class LeRobotConfig:
    """Small config object for online-first LeRobot ingestion."""

    source: str
    backend: str
    streaming: bool
    seed: int
    default_repo_id: str
    robocasa_repo_id: str
    revision: Optional[str]
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
        revision=payload.get("revision"),
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
    revision: Optional[str] = None,
) -> Any:
    """Create a LeRobot dataset object using optional runtime imports."""

    if streaming:
        module = importlib.import_module("lerobot.datasets.streaming_dataset")
        dataset_cls = getattr(module, "StreamingLeRobotDataset")
        return dataset_cls(repo_id, revision=revision)

    module = importlib.import_module("lerobot.datasets.lerobot_dataset")
    dataset_cls = getattr(module, "LeRobotDataset")
    if root is None:
        return dataset_cls(repo_id, revision=revision)
    return dataset_cls(repo_id, root=root, revision=revision)


def configured_repo_id(config: LeRobotConfig, *, source_family: str = "lerobot") -> str:
    if source_family == "robocasa":
        return config.robocasa_repo_id
    return config.default_repo_id


def first_present(sample: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in sample:
            return sample[key]
    return None


def sample_to_step(
    sample: Dict[str, Any],
    *,
    t: int,
    config: LeRobotConfig,
    repo_id: str,
    task_id: Optional[str] = None,
    instruction: Optional[str] = None,
    done: bool = False,
) -> Step:
    """Normalize one LeRobot sample dictionary into a canonical step."""

    rgb = _normalize_image(first_present(sample, config.image_keys))
    proprio = _normalize_vector(first_present(sample, config.state_keys), fallback_dim=1)
    action_vector = _normalize_vector(first_present(sample, config.action_keys), fallback_dim=1)
    language = _extract_language(sample, fallback=instruction or config.instruction)

    observation = Observation(
        rgb=rgb,
        depth=None,
        proprio=proprio,
        language=language,
        camera_intrinsics=None,
        camera_extrinsics=None,
        sim_state={
            "repo_id": repo_id,
            "task_id": task_id or config.task_id,
            "frame_index": _scalar(sample.get("frame_index", t)),
            "episode_index": _scalar(sample.get("episode_index", 0)),
            "timestamp": _scalar(sample.get("timestamp", t)),
        },
    )
    return Step(
        t=t,
        observation=observation,
        action=Action(vector=action_vector, convention=config.action_convention),
        reward=_optional_float(sample.get("next.reward", sample.get("reward"))),
        done=done,
        info={
            "source": "lerobot",
            "repo_id": repo_id,
            "task_index": _scalar(sample.get("task_index", 0)),
            "raw_keys": sorted(str(key) for key in sample.keys()),
        },
    )


def samples_to_episode(
    samples: Sequence[Dict[str, Any]],
    *,
    episode_id: str,
    config: LeRobotConfig,
    repo_id: str,
    source: str = "lerobot",
    task_id: Optional[str] = None,
    instruction: Optional[str] = None,
) -> Episode:
    """Build one canonical episode from a bounded sequence of LeRobot samples."""

    if not samples:
        raise ValueError("cannot create an episode from zero samples")
    resolved_task = task_id or _extract_task_id(samples[0], fallback=config.task_id)
    resolved_instruction = instruction or _extract_language(samples[0], fallback=config.instruction)
    steps = [
        sample_to_step(
            sample,
            t=t,
            config=config,
            repo_id=repo_id,
            task_id=resolved_task,
            instruction=resolved_instruction,
            done=t == len(samples) - 1,
        )
        for t, sample in enumerate(samples)
    ]
    return Episode(
        episode_id=episode_id,
        source="robocasa" if source == "robocasa" else "lerobot",
        task_id=resolved_task,
        instruction=resolved_instruction,
        steps=steps,
        success=None,
        metadata={
            "backend": config.backend,
            "repo_id": repo_id,
            "streaming": config.streaming,
            "source_family": source,
            "episode_index": _scalar(samples[0].get("episode_index", 0)),
        },
    )


def collect_episode_samples(
    dataset: Any,
    *,
    limit_steps: int,
    episode_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Collect a bounded episode-like sample list from a LeRobot dataset iterator."""

    collected: List[Dict[str, Any]] = []
    selected_episode = episode_index
    for raw in dataset:
        sample = dict(raw)
        sample_episode = _optional_int(sample.get("episode_index"))
        if selected_episode is None and sample_episode is not None:
            selected_episode = sample_episode
        if selected_episode is not None and sample_episode is not None and sample_episode != selected_episode:
            if collected:
                break
            continue
        collected.append(sample)
        if len(collected) >= limit_steps:
            break
    return collected


def make_fake_online_samples(*, steps: int, image_size: int = 32, episode_index: int = 0) -> List[Dict[str, Any]]:
    """Create Hub-shaped samples for offline tests of the online ingestion path."""

    samples: List[Dict[str, Any]] = []
    for t in range(steps):
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        image[:, :] = np.array([32, 36, 44], dtype=np.uint8)
        block_x = min(image_size - 8, 4 + t * 2)
        image[10:18, block_x : block_x + 8] = np.array([220, 80, 40], dtype=np.uint8)
        samples.append(
            {
                "observation.image": image,
                "observation.state": np.linspace(0.0, 1.0, num=6, dtype=np.float32) + t,
                "action": np.linspace(-0.1, 0.1, num=4, dtype=np.float32),
                "task": "online fake LeRobot sample",
                "episode_index": episode_index,
                "frame_index": t,
                "timestamp": float(t),
            }
        )
    return samples


def _module_has_class(module_name: str, class_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    return hasattr(module, class_name)


def _normalize_image(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    arr = np.asarray(value)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], repeats=3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3:
        raise ValueError(f"LeRobot image must normalize to HWC, got shape {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, repeats=3, axis=-1)
    if arr.dtype.kind == "f":
        arr = arr * 255.0 if float(np.nanmax(arr)) <= 1.0 else arr
    return np.clip(arr[..., :3], 0, 255).astype(np.uint8)


def _normalize_vector(value: Any, *, fallback_dim: int) -> np.ndarray:
    if value is None:
        return np.zeros(fallback_dim, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1).astype(np.float32)


def _extract_language(sample: Dict[str, Any], *, fallback: str) -> str:
    for key in ("task", "language", "instruction", "episode.task"):
        value = sample.get(key)
        if value is not None:
            text = str(_scalar(value)).strip()
            if text:
                return text
    return fallback


def _extract_task_id(sample: Dict[str, Any], *, fallback: str) -> str:
    for key in ("task", "task_id", "episode.task"):
        value = sample.get(key)
        if value is not None:
            text = str(_scalar(value)).strip()
            if text:
                return text.replace(" ", "_")
    return fallback


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(_scalar(value))


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(_scalar(value))


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


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
