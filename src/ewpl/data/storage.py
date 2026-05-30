"""Filesystem storage for canonical robotics episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image

from ewpl.data.episode import episode_from_dict, episode_to_dict
from ewpl.data.schemas import Episode


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _image_from_array(rgb: np.ndarray) -> Image.Image:
    arr = np.asarray(rgb)
    if arr.ndim != 3:
        raise ValueError(f"rgb frames must be 3D arrays, got shape {arr.shape}")
    if arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _array_from_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


class CanonicalEpisodeStorage:
    """Read and write canonical episodes in a simple inspectable directory layout."""

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        self.episodes_dir = self.root / "episodes"
        self.videos_dir = self.root / "videos"

    def save_dataset_card(self, payload: Dict[str, Any]) -> None:
        _write_json(self.root / "dataset_card.json", payload)

    def load_dataset_card(self) -> Dict[str, Any]:
        return _read_json(self.root / "dataset_card.json")

    def save_episode(self, episode: Episode) -> Path:
        episode_dir = self.episodes_dir / episode.episode_id
        frames_dir = episode_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        payload = episode_to_dict(episode)
        actions = []
        proprio = []
        rewards = []
        rgb_paths = []
        depth_values = []

        for step_payload, step in zip(payload["steps"], episode.steps):
            frame_name = f"{step.t:06d}.png"
            frame_path = frames_dir / frame_name
            if isinstance(step.observation.rgb, str):
                rgb_paths.append(step.observation.rgb)
            else:
                _image_from_array(step.observation.rgb).save(frame_path)
                rgb_paths.append(str(Path("frames") / frame_name))
            step_payload["observation"]["rgb"] = rgb_paths[-1]

            actions.append(step.action.vector)
            proprio.append(step.observation.proprio)
            rewards.append(np.nan if step.reward is None else step.reward)

            if isinstance(step.observation.depth, np.ndarray):
                depth_values.append(step.observation.depth.astype(np.float32))
                step_payload["observation"]["depth"] = f"depth:{len(depth_values) - 1}"

        np.savez_compressed(
            episode_dir / "arrays.npz",
            actions=np.stack(actions).astype(np.float32),
            proprio=np.stack(proprio).astype(np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            depth=np.asarray(depth_values, dtype=np.float32) if depth_values else np.empty((0,)),
        )
        _write_json(episode_dir / "episode.json", payload)
        self._write_metadata_index()
        return episode_dir

    def load_episode(self, episode_id: str) -> Episode:
        episode_dir = self.episodes_dir / episode_id
        payload = _read_json(episode_dir / "episode.json")
        arrays = np.load(episode_dir / "arrays.npz")
        depth = arrays["depth"]

        for idx, step_payload in enumerate(payload["steps"]):
            obs_payload = step_payload["observation"]
            rgb_value = obs_payload["rgb"]
            if isinstance(rgb_value, str) and not Path(rgb_value).is_absolute():
                obs_payload["rgb"] = _array_from_image(episode_dir / rgb_value)
            obs_payload["proprio"] = arrays["proprio"][idx]
            step_payload["action"]["vector"] = arrays["actions"][idx]

            depth_value = obs_payload.get("depth")
            if isinstance(depth_value, str) and depth_value.startswith("depth:"):
                depth_idx = int(depth_value.split(":", maxsplit=1)[1])
                obs_payload["depth"] = depth[depth_idx]

        return episode_from_dict(payload)

    def list_episode_ids(self) -> List[str]:
        if not self.episodes_dir.exists():
            return []
        return sorted(path.name for path in self.episodes_dir.iterdir() if path.is_dir())

    def load_all(self) -> List[Episode]:
        return [self.load_episode(episode_id) for episode_id in self.list_episode_ids()]

    def _write_metadata_index(self) -> None:
        rows = []
        for episode_id in self.list_episode_ids():
            payload = _read_json(self.episodes_dir / episode_id / "episode.json")
            rows.append(
                {
                    "episode_id": payload["episode_id"],
                    "source": payload["source"],
                    "task_id": payload["task_id"],
                    "instruction": payload["instruction"],
                    "steps": len(payload["steps"]),
                    "success": payload["success"],
                }
            )
        _write_json(self.root / "metadata.json", {"episodes": rows})


__all__ = ["CanonicalEpisodeStorage"]
