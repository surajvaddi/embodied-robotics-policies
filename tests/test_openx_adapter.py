import json

from ewpl.data.lerobot_adapter import (
    load_lerobot_config,
    make_fake_online_samples,
    samples_to_episode,
)
from ewpl.data.storage import CanonicalEpisodeStorage
from ewpl.data.validation import validate_dataset, write_validation_report


def test_openx_lerobot_validation_report(tmp_path) -> None:
    config = load_lerobot_config("configs/data/openx_lerobot.yaml")
    episode = samples_to_episode(
        make_fake_online_samples(steps=5, episode_index=0),
        episode_id="online_00000",
        config=config,
        repo_id="lerobot/pusht",
    )
    storage = CanonicalEpisodeStorage(tmp_path / "dataset")
    storage.save_episode(episode)
    storage.save_dataset_card({"name": "test", "source": "lerobot", "episodes": 1})

    report = validate_dataset(tmp_path / "dataset")
    out = write_validation_report(report, tmp_path / "validation.json")

    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["episodes"] == 1
    assert payload["total_steps"] == 5
    assert payload["sources"] == {"lerobot": 1}
    assert payload["action_dims"] == {"4": 5}
    assert payload["proprio_dims"] == {"6": 5}
    assert payload["issues"] == []

