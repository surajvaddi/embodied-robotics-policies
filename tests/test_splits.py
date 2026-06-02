import json

import numpy as np

from ewpl.data.schemas import Action, Episode, Observation, Step
from ewpl.data.splits import (
    SplitConfig,
    assert_no_leakage,
    make_iid_split,
    make_split_manifest,
    make_task_holdout_split,
    write_split_manifest,
)


def make_episode(idx: int, task_id: str = None, source: str = "synthetic") -> Episode:
    task = task_id or f"task_{idx % 3}"
    return Episode(
        episode_id=f"ep_{idx:03d}",
        source=source,
        task_id=task,
        instruction=f"do {task}",
        steps=[
            Step(
                t=0,
                observation=Observation(
                    rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                    depth=None,
                    proprio=np.zeros(2, dtype=np.float32),
                    language=f"do {task}",
                ),
                action=Action(vector=np.zeros(2, dtype=np.float32), convention="test"),
                reward=None,
                done=True,
                info={},
            )
        ],
        success=None,
        metadata={},
    )


def test_iid_split_has_no_episode_leakage() -> None:
    episodes = [make_episode(idx) for idx in range(10)]
    splits = make_iid_split(episodes, SplitConfig(seed=3))

    assert len(splits["train"]) == 7
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 2
    assert_no_leakage(splits)


def test_task_holdout_split_holds_out_whole_tasks() -> None:
    episodes = [make_episode(idx, task_id=f"task_{idx // 2}") for idx in range(8)]
    splits = make_task_holdout_split(episodes, SplitConfig(seed=0, task_holdout_fraction=0.25))
    held_out_ids = set(splits["test"])

    assert held_out_ids
    assert_no_leakage(splits)
    held_out_tasks = {episode.task_id for episode in episodes if episode.episode_id in held_out_ids}
    train_tasks = {episode.task_id for episode in episodes if episode.episode_id in splits["train"]}
    assert held_out_tasks.isdisjoint(train_tasks)


def test_split_manifest_roundtrip(tmp_path) -> None:
    episodes = [make_episode(idx) for idx in range(5)]
    manifest = make_split_manifest(episodes, config=SplitConfig(seed=1), split_type="iid")
    path = write_split_manifest(manifest, tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["split_type"] == "iid"
    assert sum(payload["counts"].values()) == 5

