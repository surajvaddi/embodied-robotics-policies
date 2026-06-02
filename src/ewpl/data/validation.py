"""Dataset QA checks for canonical robotics datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from ewpl.data.canonical_dataset import CanonicalDataset


def validate_dataset(dataset_root: Union[str, Path]) -> Dict[str, Any]:
    """Compute lightweight completeness and shape checks for a canonical dataset."""

    dataset = CanonicalDataset(dataset_root)
    report: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "episodes": len(dataset),
        "total_steps": 0,
        "sources": {},
        "tasks": {},
        "action_dims": {},
        "proprio_dims": {},
        "image_shapes": {},
        "missing_rgb_steps": 0,
        "missing_action_steps": 0,
        "missing_proprio_steps": 0,
        "language_labels": {},
        "issues": [],
    }

    for episode in dataset.episodes():
        _count(report["sources"], episode.source)
        _count(report["tasks"], episode.task_id)
        _count(report["language_labels"], episode.instruction)
        for step in episode.steps:
            report["total_steps"] += 1
            if step.observation.rgb is None:
                report["missing_rgb_steps"] += 1
            else:
                _count(report["image_shapes"], str(tuple(np.asarray(step.observation.rgb).shape)))
            if step.action.vector is None or step.action.vector.size == 0:
                report["missing_action_steps"] += 1
            else:
                _count(report["action_dims"], str(int(step.action.vector.size)))
            if step.observation.proprio is None or step.observation.proprio.size == 0:
                report["missing_proprio_steps"] += 1
            else:
                _count(report["proprio_dims"], str(int(step.observation.proprio.size)))

    if report["episodes"] == 0:
        report["issues"].append("dataset has no episodes")
    if report["missing_rgb_steps"]:
        report["issues"].append("one or more steps are missing RGB observations")
    if report["missing_action_steps"]:
        report["issues"].append("one or more steps are missing action vectors")
    if report["missing_proprio_steps"]:
        report["issues"].append("one or more steps are missing proprio vectors")

    return report


def write_validation_report(report: Dict[str, Any], out: Union[str, Path]) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def _count(counter: Dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = validate_dataset(args.dataset)
    out = write_validation_report(report, args.out)
    print(f"Wrote validation report to {out}")
    print(f"Episodes: {report['episodes']}")
    print(f"Total steps: {report['total_steps']}")
    print(f"Issues: {len(report['issues'])}")


if __name__ == "__main__":
    main()

