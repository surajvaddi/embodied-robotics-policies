#!/usr/bin/env python
"""Create a small synthetic canonical robotics dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Union

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ewpl.data.schemas import Action, Episode, Observation, Step
from ewpl.data.storage import CanonicalEpisodeStorage
from ewpl.data.visualization import render_episode_grid


def make_episode(episode_idx: int, steps: int, rng: np.random.Generator) -> Episode:
    instruction = "put the red block on the blue plate"
    episode_steps = []
    base_color = rng.integers(24, 220, size=3, dtype=np.uint8)
    for t in range(steps):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[:, :] = base_color
        block_x = min(56, 6 + t * 3)
        block_y = 20 + episode_idx % 12
        rgb[block_y : block_y + 8, block_x : block_x + 8] = np.array([230, 40, 40], dtype=np.uint8)
        rgb[42:54, 42:56] = np.array([40, 80, 220], dtype=np.uint8)

        proprio = rng.normal(size=7).astype(np.float32)
        action_vector = rng.normal(scale=0.1, size=7).astype(np.float32)
        episode_steps.append(
            Step(
                t=t,
                observation=Observation(
                    rgb=rgb,
                    depth=None,
                    proprio=proprio,
                    language=instruction,
                    camera_intrinsics={"fx": 64.0, "fy": 64.0, "cx": 32.0, "cy": 32.0},
                    camera_extrinsics={"frame": "synthetic_world"},
                    sim_state={"object_x": float(block_x), "object_y": float(block_y)},
                ),
                action=Action(vector=action_vector, convention="delta_ee_pose_gripper"),
                reward=1.0 if t == steps - 1 else 0.0,
                done=t == steps - 1,
                info={"phase": "approach" if t < steps // 2 else "place"},
            )
        )

    return Episode(
        episode_id=f"synthetic_{episode_idx:05d}",
        source="synthetic",
        task_id="synthetic_block_place",
        instruction=instruction,
        steps=episode_steps,
        success=episode_idx % 2 == 0,
        metadata={"generator": "scripts/create_synthetic_dataset.py"},
    )


def create_dataset(out: Union[str, Path], episodes: int, steps: int, seed: int) -> Path:
    root = Path(out)
    storage = CanonicalEpisodeStorage(root)
    rng = np.random.default_rng(seed)

    for episode_idx in range(episodes):
        storage.save_episode(make_episode(episode_idx, steps, rng))

    storage.save_dataset_card(
        {
            "name": "synthetic_smoke",
            "source": "synthetic",
            "episodes": episodes,
            "steps_per_episode": steps,
            "action_dim": 7,
            "proprio_dim": 7,
            "seed": seed,
        }
    )
    render_episode_grid(root, "artifacts/plots/smoke_contact_sheet.png", max_episodes=episodes)
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = create_dataset(args.out, args.episodes, args.steps, args.seed)
    print(f"Wrote synthetic dataset to {out}")


if __name__ == "__main__":
    main()
