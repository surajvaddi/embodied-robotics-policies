from ewpl.data.lerobot_adapter import (
    check_lerobot_availability,
    configured_repo_id,
    first_present,
    load_lerobot_config,
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

