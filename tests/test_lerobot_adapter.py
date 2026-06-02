from ewpl.data.lerobot_adapter import (
    check_lerobot_availability,
    collect_episode_samples,
    configured_repo_id,
    first_present,
    load_lerobot_config,
    samples_to_episode,
    sample_to_step,
)


def test_lerobot_config_loads_online_repos() -> None:
    config = load_lerobot_config("configs/data/openx_lerobot.yaml")

    assert config.source == "lerobot"
    assert config.backend == "huggingface"
    assert config.streaming is True
    assert config.default_repo_id == "lerobot/pusht"
    assert config.robocasa_repo_id == "pepijn223/robocasa_CloseFridge"
    assert configured_repo_id(config) == "lerobot/pusht"
    assert configured_repo_id(config, source_family="robocasa") == "pepijn223/robocasa_CloseFridge"


def test_lerobot_availability_check_is_import_safe() -> None:
    availability = check_lerobot_availability()

    assert isinstance(availability.lerobot_installed, bool)
    assert isinstance(availability.streaming_available, bool)
    assert isinstance(availability.local_dataset_available, bool)
    assert availability.detail


def test_first_present_selects_candidate_key() -> None:
    sample = {"observation.state": [1, 2], "action": [0, 0]}

    assert first_present(sample, ["missing", "observation.state"]) == [1, 2]
    assert first_present(sample, ["missing"]) is None


def test_lerobot_sample_to_step_normalizes_canonical_fields() -> None:
    config = load_lerobot_config("configs/data/openx_lerobot.yaml")
    sample = {
        "observation.image": [[[0, 0, 0], [255, 0, 0]], [[0, 255, 0], [0, 0, 255]]],
        "observation.state": [1.0, 2.0, 3.0],
        "action": [0.1, 0.2],
        "task": "push the T block",
        "episode_index": 7,
        "frame_index": 3,
    }

    step = sample_to_step(
        sample,
        t=0,
        config=config,
        repo_id="lerobot/pusht",
        done=True,
    )

    assert step.observation.rgb.shape == (2, 2, 3)
    assert step.observation.proprio.shape == (3,)
    assert step.action.vector.shape == (2,)
    assert step.observation.language == "push the T block"
    assert step.done is True
    assert step.info["repo_id"] == "lerobot/pusht"


def test_lerobot_samples_to_episode_preserves_hub_metadata() -> None:
    config = load_lerobot_config("configs/data/openx_lerobot.yaml")
    samples = [
        {
            "observation.image": [[[0, 0, 0]]],
            "observation.state": [1.0],
            "action": [0.1],
            "task": "close fridge",
            "episode_index": 2,
            "frame_index": idx,
        }
        for idx in range(3)
    ]

    episode = samples_to_episode(
        samples,
        episode_id="online_00000",
        config=config,
        repo_id="pepijn223/robocasa_CloseFridge",
        source="robocasa",
    )

    assert episode.source == "robocasa"
    assert episode.task_id == "close_fridge"
    assert episode.metadata["repo_id"] == "pepijn223/robocasa_CloseFridge"
    assert len(episode.steps) == 3
    assert episode.steps[-1].done is True


def test_collect_episode_samples_bounds_iterator() -> None:
    dataset = [
        {"episode_index": 0, "frame_index": 0},
        {"episode_index": 0, "frame_index": 1},
        {"episode_index": 1, "frame_index": 0},
    ]

    samples = collect_episode_samples(dataset, limit_steps=8)

    assert [sample["frame_index"] for sample in samples] == [0, 1]
