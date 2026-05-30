#!/usr/bin/env python
"""Run smoke rollouts for supported environments and policies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ewpl.sim.rollout import run_random_rollouts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=["libero", "robocasa"])
    parser.add_argument("--policy", required=True, choices=["random"])
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=12)
    args = parser.parse_args()

    metrics_path, metrics = run_random_rollouts(
        env_name=args.env,
        tasks=args.tasks,
        episodes=args.episodes,
        out=args.out,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    success_rate = sum(1 for row in metrics if row["success"]) / len(metrics)

    print(f"rollout_metrics.csv: {metrics_path}")
    print(f"episodes: {len(metrics)}")
    print(f"success_rate: {success_rate:.3f}")


if __name__ == "__main__":
    main()
