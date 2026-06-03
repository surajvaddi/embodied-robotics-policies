import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ewpl.data.schemas import Action, Episode, Observation, Step
from ewpl.data.storage import CanonicalEpisodeStorage
from ewpl.data.torch_dataset import CanonicalStepDataset


def make_episode(episode_id: str, *, steps: int = 3) -> Episode:
    transitions = []
    for t in range(steps):
        transitions.append(
            Step(
                t=t,
                observation=Observation(
                    rgb=np.full((8, 8, 3), t, dtype=np.uint8),
                    depth=None,
                    proprio=np.ones(7, dtype=np.float32) * t,
                    language="open the drawer",
                ),
                action=Action(
                    vector=np.ones(4, dtype=np.float32) * (t + 1),
                    convention="delta_ee_pose_gripper",
                ),
                reward=float(t),
                done=t == steps - 1,
                info={},
            )
        )
    return Episode(
        episode_id=episode_id,
        source="synthetic",
        task_id="drawer",
        instruction="open the drawer",
        steps=transitions,
        success=True,
    )


def test_canonical_step_dataset_flattens_episodes(tmp_path) -> None:
    storage = CanonicalEpisodeStorage(tmp_path)
    storage.save_episode(make_episode("ep0", steps=2))
    storage.save_episode(make_episode("ep1", steps=3))

    dataset = CanonicalStepDataset(tmp_path, image_size=16, language_length=6)

    assert len(dataset) == 5
    sample = dataset[3]
    assert sample["image"].shape == (3, 16, 16)
    assert sample["proprio"].shape == (7,)
    assert sample["language_ids"].shape == (6,)
    assert sample["action"].shape == (4,)
    assert sample["episode_id"] == "ep1"
    assert sample["step_index"] == 1


def test_canonical_step_dataset_batches_tensors(tmp_path) -> None:
    storage = CanonicalEpisodeStorage(tmp_path)
    storage.save_episode(make_episode("ep0", steps=4))

    dataset = CanonicalStepDataset(tmp_path, image_size=8, language_length=5)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))

    assert batch["image"].shape == (2, 3, 8, 8)
    assert batch["proprio"].shape == (2, 7)
    assert batch["language_ids"].shape == (2, 5)
    assert batch["action"].shape == (2, 4)
