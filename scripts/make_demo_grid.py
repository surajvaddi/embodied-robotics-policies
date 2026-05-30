#!/usr/bin/env python
"""Render a grid of canonical demonstration frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ewpl.data.visualization import render_episode_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_episodes", type=int, default=8)
    args = parser.parse_args()

    out = render_episode_grid(args.dataset, args.out, max_episodes=args.max_episodes)
    print(f"Wrote demo grid to {out}")


if __name__ == "__main__":
    main()

