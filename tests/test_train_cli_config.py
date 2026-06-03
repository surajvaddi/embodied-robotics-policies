from scripts.train_bc import _resolve_config


def test_train_cli_config_file_loads() -> None:
    config = _resolve_config({"config": "configs/train/bc_libero_smoke.yaml"})

    assert config["dataset_root"] == "data/smoke"
    assert config["output_dir"] == "artifacts/training/bc_libero_smoke"
    assert config["max_steps"] == 20


def test_train_cli_args_override_config() -> None:
    config = _resolve_config(
        {
            "config": "configs/train/bc_libero_smoke.yaml",
            "dataset_root": "data/custom",
            "max_steps": 3,
            "batch_size": 2,
        }
    )

    assert config["dataset_root"] == "data/custom"
    assert config["max_steps"] == 3
    assert config["batch_size"] == 2
