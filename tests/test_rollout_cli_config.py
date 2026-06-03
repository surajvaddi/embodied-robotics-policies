from scripts.eval_rollouts import _resolve_config


def test_rollout_cli_config_file_loads() -> None:
    config = _resolve_config({"config": "configs/eval/rollout_libero.yaml"})

    assert config["env"] == "libero"
    assert config["policy"] == "random"
    assert config["episodes"] == 5


def test_rollout_cli_args_override_config() -> None:
    config = _resolve_config(
        {
            "config": "configs/eval/rollout_libero.yaml",
            "episodes": 2,
            "out": "custom/out",
        }
    )

    assert config["episodes"] == 2
    assert config["out"] == "custom/out"

