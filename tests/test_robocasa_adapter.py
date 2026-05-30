import numpy as np

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

