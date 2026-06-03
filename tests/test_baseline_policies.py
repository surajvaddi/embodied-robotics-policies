import numpy as np

from ewpl.data.schemas import Observation
from ewpl.models.policies.baselines import RandomPolicy, ReplayPolicy, make_policy


def observation(dim: int = 4) -> Observation:
    return Observation(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=None,
        proprio=np.ones(dim, dtype=np.float32),
        language="test",
    )


def test_random_policy_is_seed_deterministic() -> None:
    obs = observation(dim=5)
    a = RandomPolicy(seed=7).act(obs).vector
    b = RandomPolicy(seed=7).act(obs).vector

    assert a.shape == (5,)
    assert np.allclose(a, b)


def test_replay_policy_repeats_final_action() -> None:
    policy = ReplayPolicy([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
    obs = observation(dim=2)

    assert np.allclose(policy.act(obs).vector, [1.0, 2.0])
    assert np.allclose(policy.act(obs).vector, [3.0, 4.0])
    assert np.allclose(policy.act(obs).vector, [3.0, 4.0])


def test_make_policy_returns_random_policy() -> None:
    policy = make_policy("random", seed=3, action_scale=0.2)

    assert isinstance(policy, RandomPolicy)

