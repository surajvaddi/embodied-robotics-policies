import json

import numpy as np

from ewpl.data.schemas import Action, Episode, Observation, Step
from ewpl.eval.statistics import compute_dataset_statistics, write_statistics


def make_episode(idx: int) -> Episode:
    steps = []
    for t in range(idx + 1):
        steps.append(
            Step(
                t=t,
                observation=Observation(
                    rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                    depth=None,
                    proprio=np.ones(3, dtype=np.float32),
                    language="pick up cube",
                ),
                action=Action(vector=np.ones(2, dtype=np.float32) * t, convention="test"),
                reward=None,
                done=t == idx,
                info={},
            )
        )
    return Episode(
        episode_id=f"ep_{idx}",
        source="synthetic",
        task_id="pick_cube",
        instruction="pick up cube",
        steps=steps,
        success=True,
        metadata={},
    )


def test_compute_dataset_statistics() -> None:
    stats = compute_dataset_statistics([make_episode(0), make_episode(2)])

    assert stats["episodes"] == 2
    assert stats["total_steps"] == 4
    assert stats["sources"] == {"synthetic": 2}
    assert stats["tasks"] == {"pick_cube": 2}
    assert stats["action_dims"] == {"2": 4}
    assert stats["proprio_dims"] == {"3": 4}
    assert stats["steps_per_episode_summary"]["max"] == 3.0


def test_write_statistics(tmp_path) -> None:
    stats = compute_dataset_statistics([make_episode(1)])
    path = write_statistics(stats, tmp_path / "stats.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["episodes"] == 1
    assert payload["total_steps"] == 2

