import numpy as np

from ewpl.data.schemas import Action, Observation
from ewpl.models.policies.base import BasePolicy


class ConstantPolicy(BasePolicy):
    def act(self, observation: Observation) -> Action:
        self.state.step += 1
        return Action(vector=np.zeros_like(observation.proprio), convention="constant")


def test_base_policy_reset_and_state() -> None:
    policy = ConstantPolicy()
    observation = Observation(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=None,
        proprio=np.ones(3, dtype=np.float32),
        language="test",
    )

    action = policy.act(observation)

    assert action.vector.shape == (3,)
    assert policy.state.step == 1
    assert policy.action_chunk(observation, horizon=4) is None

    policy.reset()

    assert policy.state.step == 0

