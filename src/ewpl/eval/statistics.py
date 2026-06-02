"""Dataset statistics for canonical robotics datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Union

import numpy as np

from ewpl.data.canonical_dataset import CanonicalDataset
from ewpl.data.schemas import Episode


def compute_dataset_statistics(dataset: Union[CanonicalDataset, Iterable[Episode]]) -> Dict[str, Any]:
    episodes = dataset.episodes() if isinstance(dataset, CanonicalDataset) else list(dataset)
    stats: Dict[str, Any] = {
        "episodes": len(episodes),
        "total_steps": 0,
        "sources": {},
        "tasks": {},
        "instructions": {},
        "steps_per_episode": [],
        "action_dims": {},
        "proprio_dims": {},
        "image_shapes": {},
        "action_l2": [],
        "language_lengths": [],
    }

    for episode in episodes:
        _count(stats["sources"], episode.source)
        _count(stats["tasks"], episode.task_id)
        _count(stats["instructions"], episode.instruction)
        stats["steps_per_episode"].append(len(episode.steps))
        stats["language_lengths"].append(len(episode.instruction.split()))
        for step in episode.steps:
            stats["total_steps"] += 1
            _count(stats["action_dims"], str(int(step.action.vector.size)))
            _count(stats["proprio_dims"], str(int(step.observation.proprio.size)))
            _count(stats["image_shapes"], str(tuple(np.asarray(step.observation.rgb).shape)))
            stats["action_l2"].append(float(np.linalg.norm(step.action.vector)))

    stats["steps_per_episode_summary"] = _summary(stats["steps_per_episode"])
    stats["action_l2_summary"] = _summary(stats["action_l2"])
    stats["language_length_summary"] = _summary(stats["language_lengths"])
    return stats


def write_statistics(stats: Dict[str, Any], out: Union[str, Path]) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _summary(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _count(counter: Dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    stats = compute_dataset_statistics(CanonicalDataset(args.dataset))
    path = write_statistics(stats, args.out)
    print(f"Wrote dataset statistics to {path}")
    print(f"Episodes: {stats['episodes']}")
    print(f"Total steps: {stats['total_steps']}")


if __name__ == "__main__":
    main()

