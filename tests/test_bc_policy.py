import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ewpl.data.schemas import Observation
from ewpl.models.policies.bc import BCPolicy


def test_bc_policy_forward_shape() -> None:
    policy = BCPolicy(proprio_dim=7, action_dim=4)
    batch = {
        "image": torch.zeros(3, 3, 32, 32),
        "language_ids": torch.ones(3, 16, dtype=torch.long),
        "proprio": torch.zeros(3, 7),
    }

    action = policy(batch)

    assert action.shape == (3, 4)


def test_bc_policy_act_returns_canonical_action() -> None:
    policy = BCPolicy(proprio_dim=7, action_dim=4)
    observation = Observation(
        rgb=np.zeros((32, 32, 3), dtype=np.uint8),
        depth=None,
        proprio=np.ones(7, dtype=np.float32),
        language="move cube",
    )

    action = policy.act(observation)

    assert action.vector.shape == (4,)
    assert action.convention == "delta_ee_pose_gripper"
    assert policy.state.step == 1

