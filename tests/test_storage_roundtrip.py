import numpy as np

from ewpl.data.canonical_dataset import CanonicalDataset
from ewpl.data.schemas import Action, Episode, Observation, Step
from ewpl.data.storage import CanonicalEpisodeStorage


def make_episode(episode_id: str = "ep0") -> Episode:
    steps = []
    for t in range(4):
        rgb = np.full((16, 16, 3), fill_value=t * 20, dtype=np.uint8)
        steps.append(
            Step(
                t=t,
                observation=Observation(
                    rgb=rgb,
                    depth=np.ones((16, 16), dtype=np.float32) * t,
                    proprio=np.ones(7, dtype=np.float32) * t,
                    language="pick up the cube",
                    sim_state={"t": t},
                ),
                action=Action(vector=np.ones(7, dtype=np.float32) * (t + 1), convention="delta_ee"),
                reward=float(t),
                done=t == 3,
                info={"idx": t},
            )
        )
    return Episode(
        episode_id=episode_id,
        source="synthetic",
        task_id="task_pick",
        instruction="pick up the cube",
        steps=steps,
        success=True,
        metadata={"split": "smoke"},
    )


def test_storage_roundtrip(tmp_path) -> None:
    storage = CanonicalEpisodeStorage(tmp_path)
    storage.save_dataset_card({"name": "roundtrip"})
    storage.save_episode(make_episode())

    restored = storage.load_episode("ep0")

    assert storage.load_dataset_card()["name"] == "roundtrip"
    assert restored.episode_id == "ep0"
    assert restored.steps[2].observation.rgb.shape == (16, 16, 3)
    assert np.allclose(restored.steps[2].observation.proprio, np.ones(7) * 2)
    assert np.allclose(restored.steps[2].action.vector, np.ones(7) * 3)
    assert np.allclose(restored.steps[2].observation.depth, np.ones((16, 16)) * 2)
    assert (tmp_path / "metadata.json").exists()


def test_canonical_dataset_loads_episodes(tmp_path) -> None:
    storage = CanonicalEpisodeStorage(tmp_path)
    storage.save_episode(make_episode("ep_a"))
    storage.save_episode(make_episode("ep_b"))

    dataset = CanonicalDataset(tmp_path)

    assert len(dataset) == 2
    assert dataset[0].episode_id == "ep_a"
    assert dataset[1].episode_id == "ep_b"

