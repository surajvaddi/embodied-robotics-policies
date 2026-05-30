"""Rollout helpers for smoke evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from ewpl.data.schemas import Action
from ewpl.sim.robocasa_env import RoboCasaEnv


def make_env(env_name: str):
    if env_name == "robocasa":
        return RoboCasaEnv()
    raise ValueError(f"unsupported rollout env: {env_name}")


def run_random_rollouts(
    *,
    env_name: str,
    tasks: int,
    episodes: int,
    out: str,
    seed: int = 0,
    max_steps: int = 12,
) -> Tuple[Path, List[Dict[str, object]]]:
    """Run random smoke rollouts and write per-episode metrics plus frames."""

    out_dir = Path(out)
    frames_root = out_dir / "videos"
    frames_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    metrics: List[Dict[str, object]] = []

    for episode_idx in range(episodes):
        env = make_env(env_name)
        task_id = f"smoke_task_{episode_idx % max(1, tasks):03d}"
        scene_id = f"kitchen_scene_{episode_idx % 3:03d}"
        observation = env.reset(task_id=task_id, scene_id=scene_id, seed=seed + episode_idx)
        episode_frames = frames_root / f"episode_{episode_idx:05d}"
        episode_frames.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(observation.rgb, dtype=np.uint8)).save(episode_frames / "000000.png")

        total_reward = 0.0
        done = False
        success = False
        steps_taken = 0
        action_norms = []

        for step_idx in range(max_steps):
            action_vector = rng.normal(loc=0.0, scale=0.1, size=observation.proprio.shape[0])
            action = Action(
                vector=action_vector.astype(np.float32),
                convention="delta_ee_pose_gripper",
            )
            result = env.step(action)
            steps_taken = step_idx + 1
            total_reward += result.reward
            done = result.done
            success = result.success
            action_norms.append(float(np.linalg.norm(action.vector)))
            Image.fromarray(np.asarray(result.observation.rgb, dtype=np.uint8)).save(
                episode_frames / f"{steps_taken:06d}.png"
            )
            observation = result.observation
            if done:
                break

        metrics.append(
            {
                "episode": episode_idx,
                "env": env_name,
                "task_id": task_id,
                "scene_id": scene_id,
                "steps": steps_taken,
                "success": success,
                "total_reward": total_reward,
                "done": done,
                "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
            }
        )

    metrics_path = out_dir / "rollout_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    return metrics_path, metrics

