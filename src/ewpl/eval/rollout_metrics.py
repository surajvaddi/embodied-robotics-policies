"""Metrics computed from rollout artifact CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np


def compute_rollout_metrics(rollout_dir: Union[str, Path]) -> Dict[str, Any]:
    root = Path(rollout_dir)
    episodes = _read_csv(root / "rollout_metrics.csv")
    steps = _read_csv(root / "per_step_logs.csv")

    success_values = [_as_bool(row["success"]) for row in episodes]
    episode_lengths = [float(row["steps"]) for row in episodes]
    action_norms = [float(row["action_norm"]) for row in steps if row.get("action_norm")]
    latencies = [float(row["policy_latency_ms"]) for row in steps if row.get("policy_latency_ms")]

    return {
        "episodes": len(episodes),
        "steps": len(steps),
        "success_rate": float(np.mean(success_values)) if success_values else 0.0,
        "average_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
        "action_smoothness": _action_smoothness(steps),
        "mean_policy_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p95_policy_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "failure_modes": _failure_modes(episodes),
    }


def write_rollout_metrics(metrics: Dict[str, Any], out: Union[str, Path]) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": json.dumps(value, sort_keys=True)})
    return path


def _action_smoothness(rows: List[Dict[str, str]]) -> float:
    by_episode: Dict[str, List[np.ndarray]] = {}
    for row in rows:
        vector = row.get("action_vector")
        if not vector:
            continue
        by_episode.setdefault(row["episode"], []).append(np.asarray(json.loads(vector), dtype=np.float32))

    diffs = []
    for vectors in by_episode.values():
        for prev, current in zip(vectors, vectors[1:]):
            diffs.append(float(np.linalg.norm(current - prev)))
    return float(np.mean(diffs)) if diffs else 0.0


def _failure_modes(rows: List[Dict[str, str]]) -> Dict[str, int]:
    modes: Dict[str, int] = {}
    for row in rows:
        if _as_bool(row.get("success", "")):
            continue
        reason = row.get("termination_reason") or "unknown"
        modes[reason] = modes.get(reason, 0) + 1
    return modes


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    metrics = compute_rollout_metrics(args.rollouts)
    path = write_rollout_metrics(metrics, args.out)
    print(f"Wrote rollout metrics to {path}")
    print(f"success_rate: {metrics['success_rate']:.3f}")
    print(f"mean_policy_latency_ms: {metrics['mean_policy_latency_ms']:.3f}")


if __name__ == "__main__":
    main()

