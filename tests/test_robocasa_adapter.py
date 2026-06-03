import json
import csv
import subprocess
import sys

import numpy as np

from ewpl.data.robocasa_adapter import index_tasks, load_robocasa_config
from ewpl.data.robocasa_adapter import convert_episode
from ewpl.data.canonical_dataset import CanonicalDataset
from ewpl.data.schemas import Action
from ewpl.data.storage import CanonicalEpisodeStorage
from ewpl.sim.rollout import run_random_rollouts
from ewpl.sim.robocasa_env import RoboCasaEnv


def test_robocasa_env_reset_step_returns_canonical_observation() -> None:
    env = RoboCasaEnv(benchmark="robocasa365")
    observation = env.reset("open_the_top_drawer", scene_id="kitchen_scene_001", seed=13)

    assert observation.rgb.shape == (80, 80, 3)
    assert observation.proprio.shape == (9,)
    assert "RoboCasa task" in observation.language
    assert observation.sim_state["scene_id"] == "kitchen_scene_001"

    result = env.step(Action(vector=np.zeros(9, dtype=np.float32), convention="delta_ee_pose_gripper"))

    assert result.observation.rgb.shape == (80, 80, 3)
    assert isinstance(result.done, bool)
    assert result.info["task_id"] == "open_the_top_drawer"
    assert env.render().shape == (80, 80, 3)
    assert env.get_task_metadata()["source"] == "robocasa"


def test_robocasa_config_and_task_indexing() -> None:
    config = load_robocasa_config("configs/data/robocasa.yaml")
    manifest = index_tasks(config, limit_tasks=3, dry_run=True)

    assert manifest["benchmark"] == "robocasa365"
    assert manifest["tasks_indexed"] == 3
    assert manifest["scene_variations_indexed"] == 3
    assert manifest["tasks"][0]["task_id"] == "open_the_top_drawer"
    assert manifest["demo_files_available"] in {True, False}


def test_download_robocasa_dry_run_script(tmp_path) -> None:
    out = tmp_path / "robocasa_manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/download_robocasa.py",
            "--dry_run",
            "--limit_tasks",
            "3",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert "RoboCasa tasks indexed: 3" in result.stdout
    assert manifest["tasks_indexed"] == 3
    assert manifest["scene_variations_indexed"] == 3


def test_robocasa_episode_conversion_roundtrip(tmp_path) -> None:
    episode = convert_episode(
        "open_the_top_drawer",
        "kitchen_scene_001",
        episode_idx=0,
        steps=6,
        seed=5,
    )
    storage = CanonicalEpisodeStorage(tmp_path)
    storage.save_episode(episode)

    dataset = CanonicalDataset(tmp_path)
    restored = dataset[0]

    assert restored.source == "robocasa"
    assert restored.task_id == "open_the_top_drawer"
    assert restored.metadata["scene_id"] == "kitchen_scene_001"
    assert len(restored.steps) == 6
    assert restored.steps[0].observation.rgb.shape == (80, 80, 3)


def test_robocasa_random_rollout_smoke(tmp_path) -> None:
    metrics_path, metrics = run_random_rollouts(
        env_name="robocasa",
        tasks=2,
        episodes=3,
        out=str(tmp_path / "rollouts"),
        seed=23,
        max_steps=8,
    )

    assert metrics_path.exists()
    assert len(metrics) == 3
    assert (tmp_path / "rollouts" / "videos" / "episode_00000" / "000000.png").exists()
    assert (tmp_path / "rollouts" / "per_step_logs.csv").exists()

    with metrics_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert set(rows[0]) == {
        "episode",
        "env",
        "policy",
        "task_id",
        "scene_id",
        "steps",
        "success",
        "total_reward",
        "done",
        "mean_action_norm",
        "mean_policy_latency_ms",
        "termination_reason",
    }
    with (tmp_path / "rollouts" / "per_step_logs.csv").open(encoding="utf-8") as handle:
        step_rows = list(csv.DictReader(handle))
    assert step_rows
    assert "policy_latency_ms" in step_rows[0]
