#!/usr/bin/env python
"""Ingest a small LeRobot/Open X subset into canonical storage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ewpl.data.lerobot_adapter import (
    check_lerobot_availability,
    collect_episode_samples,
    configured_repo_id,
    load_lerobot_config,
    make_fake_online_samples,
    make_lerobot_dataset,
    samples_to_episode,
)
from ewpl.data.storage import CanonicalEpisodeStorage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/openx_lerobot.yaml")
    parser.add_argument("--subset", default="small", choices=["small", "robocasa", "fake"])
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--limit_episodes", type=int, default=2)
    parser.add_argument("--steps_per_episode", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fake_online", action="store_true")
    parser.add_argument("--allow_indexed_download", action="store_true")
    args = parser.parse_args()

    config = load_lerobot_config(args.config)
    source_family = "robocasa" if args.subset == "robocasa" else "lerobot"
    repo_id = args.repo_id or configured_repo_id(config, source_family=source_family)
    steps = args.steps_per_episode or config.max_streamed_steps_per_episode
    use_fake = args.fake_online or args.subset == "fake"

    storage = CanonicalEpisodeStorage(args.out)
    availability = check_lerobot_availability()

    if use_fake:
        episodes = [
            samples_to_episode(
                make_fake_online_samples(steps=steps, episode_index=episode_idx),
                episode_id=f"online_{episode_idx:05d}",
                config=config,
                repo_id=repo_id,
                source=source_family,
            )
            for episode_idx in range(args.limit_episodes)
        ]
    else:
        if not availability.streaming_available and config.streaming:
            raise RuntimeError(
                "StreamingLeRobotDataset is unavailable. Install LeRobot or rerun with --fake_online."
            )
        dataset = make_lerobot_dataset(repo_id, streaming=config.streaming, revision=config.revision)
        episodes = []
        for episode_idx in range(args.limit_episodes):
            try:
                samples = collect_episode_samples(
                    dataset,
                    limit_steps=steps,
                    episode_index=episode_idx,
                )
            except Exception as exc:
                if not config.streaming:
                    raise
                print(f"Streaming read failed, falling back to indexed access: {exc}")
                samples = []
            if config.streaming and len(samples) < steps:
                if source_family == "robocasa" and not args.allow_indexed_download:
                    raise RuntimeError(
                        "RoboCasa streaming produced too few samples or failed. "
                        "Pass --allow_indexed_download to permit local indexed cache downloads."
                    )
                dataset = make_lerobot_dataset(repo_id, streaming=False, revision=config.revision)
                samples = collect_episode_samples(
                    dataset,
                    limit_steps=steps,
                    episode_index=episode_idx,
                )
            if not samples:
                break
            episodes.append(
                samples_to_episode(
                    samples,
                    episode_id=f"online_{episode_idx:05d}",
                    config=config,
                    repo_id=repo_id,
                    source=source_family,
                )
            )

    for episode in episodes:
        storage.save_episode(episode)

    storage.save_dataset_card(
        {
            "name": "online_lerobot_subset",
            "source": source_family,
            "backend": config.backend,
            "repo_id": repo_id,
            "streaming": config.streaming,
            "fake_online": use_fake,
            "episodes": len(episodes),
            "steps_per_episode": steps,
            "lerobot_installed": availability.lerobot_installed,
            "streaming_available": availability.streaming_available,
        }
    )

    print(f"Wrote canonical online dataset to {args.out}")
    print(f"Repo: {repo_id}")
    print(f"Episodes: {len(episodes)}")
    print(f"Fake online: {str(use_fake).lower()}")


if __name__ == "__main__":
    main()
