import json
import subprocess
import sys

import numpy as np

from ewpl.data.canonical_dataset import CanonicalDataset
from ewpl.data.libero_adapter import convert_episode, index_tasks, load_libero_config
from ewpl.data.storage import CanonicalEpisodeStorage
from ewpl.sim.rollout import run_random_rollouts
from ewpl.sim.libero_env import LiberoEnv


def test_libero_env_reset_step_returns_canonical_observation() -> None:
    env = LiberoEnv(suite="libero_spatial")
    observation = env.reset("put_the_bowl_on_the_plate", seed=11)

    assert observation.rgb.shape == (64, 64, 3)
    assert observation.proprio.shape == (7,)
    assert "LIBERO task" in observation.language

    result = env.step(action=convert_episode("put_the_bowl_on_the_plate", episode_idx=0).steps[0].action)

    assert result.observation.rgb.shape == (64, 64, 3)
    assert isinstance(result.done, bool)
    assert result.info["task_id"] == "put_the_bowl_on_the_plate"


def test_libero_config_and_task_indexing() -> None:
    config = load_libero_config("configs/data/libero.yaml")
    manifest = index_tasks(config, limit_tasks=2, dry_run=True)

    assert manifest["suite"] == "libero_spatial"
    assert manifest["tasks_indexed"] == 2
    assert manifest["tasks"][0]["task_id"] == "put_the_bowl_on_the_plate"
    assert manifest["demo_files_available"] in {True, False}


def test_libero_episode_conversion_roundtrip(tmp_path) -> None:
    episode = convert_episode(
        "put_the_bowl_on_the_plate",
        episode_idx=0,
        steps=5,
        seed=3,
    )
    storage = CanonicalEpisodeStorage(tmp_path)
    storage.save_episode(episode)

    dataset = CanonicalDataset(tmp_path)
    restored = dataset[0]

    assert restored.source == "libero"
    assert restored.task_id == "put_the_bowl_on_the_plate"
    assert len(restored.steps) == 5
    assert np.asarray(restored.steps[0].observation.rgb).shape == (64, 64, 3)


def test_download_libero_dry_run_script(tmp_path) -> None:
    out = tmp_path / "libero_manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/download_libero.py",
            "--suite",
            "libero_spatial",
            "--limit_tasks",
            "2",
            "--dry_run",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert "Found LIBERO suite: libero_spatial" in result.stdout
    assert manifest["tasks_indexed"] == 2


def test_libero_random_rollout_smoke(tmp_path) -> None:
    metrics_path, metrics = run_random_rollouts(
        env_name="libero",
        tasks=2,
        episodes=2,
        out=str(tmp_path / "rollouts"),
        seed=31,
        max_steps=6,
    )

    assert metrics_path.exists()
    assert len(metrics) == 2
    assert (tmp_path / "rollouts" / "videos" / "episode_00000" / "000000.png").exists()
