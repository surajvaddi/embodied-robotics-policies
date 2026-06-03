"""Rollout helpers for smoke evaluation."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ewpl.models.policies.base import Policy
from ewpl.models.policies.baselines import RandomPolicy
from ewpl.sim.libero_env import LiberoEnv
from ewpl.sim.robocasa_env import RoboCasaEnv
from ewpl.sim.video import write_frame


@dataclass
class RolloutConfig:
    env_name: str
    tasks: int
    episodes: int
    out: str
    seed: int = 0
    max_steps: int = 12
    policy_name: str = "random"


def make_env(env_name: str):
    if env_name == "libero":
        return LiberoEnv()
    if env_name == "robocasa":
        return RoboCasaEnv()
    raise ValueError(f"unsupported rollout env: {env_name}")


def run_policy_rollouts(
    *,
    policy: Policy,
    config: RolloutConfig,
) -> Tuple[Path, List[Dict[str, object]]]:
    """Run a policy and write frames, episode summaries, and per-step logs."""

    out_dir = Path(config.out)
    frames_root = out_dir / "videos"
    frames_root.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, object]] = []
    step_logs: List[Dict[str, object]] = []

    for episode_idx in range(config.episodes):
        env = make_env(config.env_name)
        task_id = f"smoke_task_{episode_idx % max(1, config.tasks):03d}"
        scene_id = f"kitchen_scene_{episode_idx % 3:03d}"
        if config.env_name == "robocasa":
            observation = env.reset(task_id=task_id, scene_id=scene_id, seed=config.seed + episode_idx)
        else:
            observation = env.reset(task_id=task_id, seed=config.seed + episode_idx)
        policy.reset()
        episode_frames = frames_root / f"episode_{episode_idx:05d}"
        write_frame(observation.rgb, episode_frames / "000000.png")

        total_reward = 0.0
        done = False
        success = False
        steps_taken = 0
        action_norms = []
        latencies_ms = []

        for step_idx in range(config.max_steps):
            start = time.perf_counter()
            action = policy.act(observation)
            latency_ms = (time.perf_counter() - start) * 1000.0
            result = env.step(action)
            steps_taken = step_idx + 1
            total_reward += result.reward
            done = result.done
            success = result.success
            action_norm = float(np.linalg.norm(action.vector))
            action_norms.append(action_norm)
            latencies_ms.append(latency_ms)
            write_frame(result.observation.rgb, episode_frames / f"{steps_taken:06d}.png")
            step_logs.append(
                {
                    "episode": episode_idx,
                    "step": step_idx,
                    "env": config.env_name,
                    "policy": config.policy_name,
                    "task_id": task_id,
                    "scene_id": scene_id if config.env_name == "robocasa" else "",
                    "reward": result.reward,
                    "done": result.done,
                    "success": result.success,
                    "action_norm": action_norm,
                    "policy_latency_ms": latency_ms,
                    "termination_reason": "success" if result.success else ("done" if result.done else ""),
                }
            )
            observation = result.observation
            if done:
                break

        summaries.append(
            {
                "episode": episode_idx,
                "env": config.env_name,
                "policy": config.policy_name,
                "task_id": task_id,
                "scene_id": scene_id if config.env_name == "robocasa" else "",
                "steps": steps_taken,
                "success": success,
                "total_reward": total_reward,
                "done": done,
                "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
                "mean_policy_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
                "termination_reason": "success" if success else ("done" if done else "max_steps"),
            }
        )

    metrics_path = out_dir / "rollout_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    step_logs_path = out_dir / "per_step_logs.csv"
    with step_logs_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(step_logs[0].keys()) if step_logs else ["episode", "step"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(step_logs)

    return metrics_path, summaries


def run_random_rollouts(
    *,
    env_name: str,
    tasks: int,
    episodes: int,
    out: str,
    seed: int = 0,
    max_steps: int = 12,
    action_scale: float = 0.1,
) -> Tuple[Path, List[Dict[str, object]]]:
    """Run random smoke rollouts and write comparable rollout artifacts."""

    config = RolloutConfig(
        env_name=env_name,
        tasks=tasks,
        episodes=episodes,
        out=out,
        seed=seed,
        max_steps=max_steps,
        policy_name="random",
    )
    policy = RandomPolicy(seed=seed, scale=action_scale)
    return run_policy_rollouts(policy=policy, config=config)
