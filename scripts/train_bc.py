#!/usr/bin/env python
"""Train a behavior cloning policy on a canonical dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/bc_libero_smoke.yaml")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--language_length", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--action_convention", default=None)
    args = parser.parse_args()
    config = _resolve_config(vars(args))

    from ewpl.training.bc import BCTrainingConfig, train_bc

    result = train_bc(BCTrainingConfig(**config))
    print(f"checkpoint: {result.checkpoint_path}")
    print(f"metrics: {result.metrics_path}")
    print(f"steps: {len(result.losses)}")
    print(f"final_loss: {result.losses[-1]:.6f}")


def _resolve_config(args: Dict[str, Any]) -> Dict[str, Any]:
    config = _load_simple_yaml(args["config"]) if args.get("config") else {}
    for key, value in args.items():
        if key == "config":
            continue
        if value is not None:
            config[key] = value

    defaults = {
        "dataset_root": "data/smoke",
        "output_dir": "artifacts/training/bc_libero_smoke",
        "max_steps": 20,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "seed": 0,
        "image_size": 32,
        "language_length": 16,
        "vocab_size": 4096,
        "hidden_dim": 64,
        "action_convention": "delta_ee_pose_gripper",
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
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
