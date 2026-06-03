"""Policy interfaces and baseline policies."""

from ewpl.models.policies.base import Policy, PolicyState
from ewpl.models.policies.baselines import RandomPolicy, ReplayPolicy, make_policy

__all__ = ["Policy", "PolicyState", "RandomPolicy", "ReplayPolicy", "BCPolicy", "make_policy"]


def __getattr__(name: str):
    if name == "BCPolicy":
        from ewpl.models.policies.bc import BCPolicy

        return BCPolicy
    raise AttributeError(name)
