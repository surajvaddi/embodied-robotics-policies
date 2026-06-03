import numpy as np
import pytest

pytest.importorskip("torch")

from ewpl.data.schemas import Action, Episode, Observation, Step
from ewpl.data.storage import CanonicalEpisodeStorage
from ewpl.training.bc import BCTrainingConfig, train_bc


def make_episode(episode_id: str, *, steps: int = 4) -> Episode:
    transitions = []
    for t in range(steps):
        transitions.append(
            Step(
                t=t,
                observation=Observation(
                    rgb=np.full((12, 12, 3), 120 + t, dtype=np.uint8),
                    depth=None,
                    proprio=np.full(7, t / 10, dtype=np.float32),
                    language="move to target",
                ),
                action=Action(
                    vector=np.full(4, 0.1 * (t + 1), dtype=np.float32),
                    convention="delta_ee_pose_gripper",
                ),
                reward=None,
                done=t == steps - 1,
                info={},
            )
        )
    return Episode(
        episode_id=episode_id,
        source="synthetic",
        task_id="target",
        instruction="move to target",
        steps=transitions,
        success=True,
    )


def test_train_bc_writes_metrics_and_checkpoint(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    storage = CanonicalEpisodeStorage(dataset_root)
    storage.save_episode(make_episode("ep0"))
    storage.save_episode(make_episode("ep1"))

    result = train_bc(
        BCTrainingConfig(
            dataset_root=dataset_root,
            output_dir=tmp_path / "run",
            max_steps=3,
            batch_size=2,
            image_size=16,
            language_length=5,
            hidden_dim=16,
            seed=1,
        )
    )

    assert len(result.losses) == 3
    assert result.metrics_path.exists()
    assert result.checkpoint_path.exists()
    assert result.metrics_path.read_text(encoding="utf-8").splitlines()[0] == "step,loss"
