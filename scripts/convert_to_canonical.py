#!/usr/bin/env python
"""Convert supported robotics sources into the canonical dataset format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ewpl.data.libero_adapter import convert_to_canonical, load_libero_config
from ewpl.data.storage import CanonicalEpisodeStorage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["libero"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit_episodes", type=int, default=20)
    args = parser.parse_args()

    if args.source != "libero":
        raise ValueError(f"unsupported source: {args.source}")

    episodes = convert_to_canonical(
        args.config,
        args.out,
        limit_episodes=args.limit_episodes,
    )
    storage = CanonicalEpisodeStorage(args.out)
    for episode in episodes:
        storage.save_episode(episode)

    config = load_libero_config(args.config)
    storage.save_dataset_card(
        {
            "name": f"{config.get('suite', 'libero')}_canonical",
            "source": "libero",
            "suite": config.get("suite", "libero_spatial"),
            "episodes": len(episodes),
            "conversion": "smoke",
        }
    )
    print(f"Wrote canonical {args.source} dataset to {args.out}")
    print(f"Episodes: {len(episodes)}")


if __name__ == "__main__":
    main()

