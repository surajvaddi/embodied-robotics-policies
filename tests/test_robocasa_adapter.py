import json
import subprocess
import sys

import numpy as np

from ewpl.data.robocasa_adapter import index_tasks, load_robocasa_config
from ewpl.data.schemas import Action
from ewpl.sim.robocasa_env import RoboCasaEnv


def test_robocasa_env_reset_step_returns_canonical_observation() -> None:
    env = RoboCasaEnv(benchmark="robocasa365")
    observation = env.reset("open_the_top_drawer", scene_id="kitchen_scene_001", seed=13)

    assert observation.rgb.shape == (80, 80, 3)
    assert observation.proprio.shape == (9,)
    assert "RoboCasa task" in observation.language
    assert observation.sim_state["scene_id"] == "kitchen_scene_001"

    result = env.step(Action(vector=np.zeros(9, dtype=np.float32), convention="delta_ee_pose_gripper"))

    assert result.observation.rgb.shape == (80, 80, 3)
    assert isinstance(result.done, bool)
    assert result.info["task_id"] == "open_the_top_drawer"
    assert env.render().shape == (80, 80, 3)
    assert env.get_task_metadata()["source"] == "robocasa"


def test_robocasa_config_and_task_indexing() -> None:
    config = load_robocasa_config("configs/data/robocasa.yaml")
    manifest = index_tasks(config, limit_tasks=3, dry_run=True)

    assert manifest["benchmark"] == "robocasa365"
    assert manifest["tasks_indexed"] == 3
    assert manifest["scene_variations_indexed"] == 3
    assert manifest["tasks"][0]["task_id"] == "open_the_top_drawer"
    assert manifest["demo_files_available"] in {True, False}


def test_download_robocasa_dry_run_script(tmp_path) -> None:
    out = tmp_path / "robocasa_manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/download_robocasa.py",
            "--dry_run",
            "--limit_tasks",
            "3",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert "RoboCasa tasks indexed: 3" in result.stdout
    assert manifest["tasks_indexed"] == 3
    assert manifest["scene_variations_indexed"] == 3
