import csv

import numpy as np
import pytest

pytest.importorskip("torch")

from ewpl.data.schemas import Action, Episode, Observation, Step
from ewpl.data.storage import CanonicalEpisodeStorage
from ewpl.sim.rollout import run_checkpoint_rollouts
from ewpl.training.bc import BCTrainingConfig, train_bc
from ewpl.training.checkpointing import load_bc_policy


def make_episode() -> Episode:
    steps = []
    for t in range(3):
        steps.append(
            Step(
                t=t,
                observation=Observation(
                    rgb=np.zeros((16, 16, 3), dtype=np.uint8),
                    depth=None,
                    proprio=np.zeros(7, dtype=np.float32),
                    language="complete LIBERO task: smoke task 000",
                ),
                action=Action(
                    vector=np.zeros(7, dtype=np.float32),
                    convention="delta_ee_pose_gripper",
                ),
                reward=None,
                done=t == 2,
                info={},
            )
        )
    return Episode(
        episode_id="ep0",
        source="synthetic",
        task_id="smoke_task_000",
        instruction="complete LIBERO task: smoke task 000",
        steps=steps,
        success=True,
    )


def test_bc_checkpoint_loads_and_rolls_out(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    storage = CanonicalEpisodeStorage(dataset_root)
    storage.save_episode(make_episode())
    result = train_bc(
        BCTrainingConfig(
            dataset_root=dataset_root,
            output_dir=tmp_path / "train",
            max_steps=2,
            batch_size=1,
            image_size=16,
            language_length=6,
            hidden_dim=16,
        )
    )

    policy = load_bc_policy(result.checkpoint_path)
    assert policy.action_dim == 7

    metrics_path, rows = run_checkpoint_rollouts(
        checkpoint_path=str(result.checkpoint_path),
        env_name="libero",
        tasks=1,
        episodes=1,
        out=str(tmp_path / "rollouts"),
        max_steps=2,
    )

    assert metrics_path.exists()
    assert rows[0]["policy"] == "bc:latest.pt"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert saved[0]["policy"] == "bc:latest.pt"
