import numpy as np
import pytest

from ewpl.data.episode import episode_from_dict, episode_to_dict
from ewpl.data.schemas import Action, Episode, Observation, Step


def make_episode() -> Episode:
    steps = []
    for t in range(3):
        steps.append(
            Step(
                t=t,
                observation=Observation(
                    rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                    depth=None,
                    proprio=np.ones(7, dtype=np.float32),
                    language="move the cube",
                ),
                action=Action(vector=np.zeros(7, dtype=np.float32), convention="delta_ee"),
                reward=None,
                done=t == 2,
                info={},
            )
        )
    return Episode(
        episode_id="ep0",
        source="synthetic",
        task_id="task0",
        instruction="move the cube",
        steps=steps,
        success=True,
        metadata={},
    )


def test_episode_validates_and_serializes() -> None:
    episode = make_episode()
    restored = episode_from_dict(episode_to_dict(episode))

    assert restored.episode_id == episode.episode_id
    assert restored.source == "synthetic"
    assert len(restored.steps) == 3
    assert restored.steps[0].observation.proprio.shape == (7,)
    assert restored.steps[0].action.vector.shape == (7,)


def test_episode_requires_contiguous_steps() -> None:
    episode = make_episode()
    episode.steps[1].t = 7

    with pytest.raises(ValueError, match="contiguous"):
        Episode(
            episode_id=episode.episode_id,
            source=episode.source,
            task_id=episode.task_id,
            instruction=episode.instruction,
            steps=episode.steps,
            success=episode.success,
            metadata=episode.metadata,
        )

