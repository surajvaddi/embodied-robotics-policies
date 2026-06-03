#!/usr/bin/env python
"""Run smoke rollouts for supported environments and policies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ewpl.sim.rollout import run_checkpoint_rollouts, run_random_rollouts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--env", default=None, choices=["libero", "robocasa"])
    parser.add_argument("--policy", default=None, choices=["random", "checkpoint"])
    parser.add_argument("--checkpoint", dest="checkpoint_path", default=None)
    parser.add_argument("--tasks", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--action_scale", type=float, default=None)
    args = parser.parse_args()
    config = _resolve_config(vars(args))

    if config["policy"] == "checkpoint":
        metrics_path, metrics = run_checkpoint_rollouts(
            checkpoint_path=config["checkpoint_path"],
            env_name=config["env"],
            tasks=int(config["tasks"]),
            episodes=int(config["episodes"]),
            out=config["out"],
            seed=int(config["seed"]),
            max_steps=int(config["max_steps"]),
        )
    else:
        metrics_path, metrics = run_random_rollouts(
            env_name=config["env"],
            tasks=int(config["tasks"]),
            episodes=int(config["episodes"]),
            out=config["out"],
            seed=int(config["seed"]),
            max_steps=int(config["max_steps"]),
            action_scale=float(config["action_scale"]),
        )
    success_rate = sum(1 for row in metrics if row["success"]) / len(metrics)

    print(f"rollout_metrics.csv: {metrics_path}")
    print(f"episodes: {len(metrics)}")
    print(f"success_rate: {success_rate:.3f}")


def _resolve_config(args: Dict[str, Any]) -> Dict[str, Any]:
    config = _load_simple_yaml(args["config"]) if args.get("config") else {}
    for key, value in args.items():
        if key == "config":
            continue
        if value is not None:
            config[key] = value

    defaults = {
        "env": None,
        "policy": "random",
        "tasks": 2,
        "episodes": 5,
        "seed": 0,
        "max_steps": 12,
        "action_scale": 0.1,
        "checkpoint_path": None,
        "out": None,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)

    if config["env"] is None:
        raise ValueError("--env or config env is required")
    if config["out"] is None:
        raise ValueError("--out or config out is required")
    if config["policy"] == "checkpoint" and not config["checkpoint_path"]:
        raise ValueError("--checkpoint or config checkpoint_path is required for checkpoint policy")
    return config


def _load_simple_yaml(path: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        payload[key.strip()] = _parse_scalar(value.strip())
    return payload


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


if __name__ == "__main__":
    main()
