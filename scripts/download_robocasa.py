#!/usr/bin/env python
"""Locate or dry-run index RoboCasa demonstrations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ewpl.data.robocasa_adapter import index_tasks, load_robocasa_config, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/robocasa.yaml")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--limit_tasks", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--out", default="artifacts/logs/robocasa_manifest.json")
    args = parser.parse_args()

    config = load_robocasa_config(args.config)
    if args.benchmark:
        config["benchmark"] = args.benchmark

    manifest = index_tasks(config, limit_tasks=args.limit_tasks, dry_run=args.dry_run)
    out = write_manifest(manifest, args.out)

    print(f"RoboCasa benchmark: {manifest['benchmark']}")
    print(f"RoboCasa tasks indexed: {manifest['tasks_indexed']}")
    print(f"Scene variations indexed: {manifest['scene_variations_indexed']}")
    print(f"Available demos: {str(manifest['demo_files_available']).lower()}")
    print(f"Manifest written: {out}")


if __name__ == "__main__":
    main()

