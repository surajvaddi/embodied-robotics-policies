"""Checkpoint loading helpers for trained policies."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from ewpl.models.policies.bc import BCPolicy


def load_bc_policy(checkpoint_path: Union[str, Path], *, map_location: str = "cpu") -> BCPolicy:
    """Load a behavior cloning policy from a training checkpoint."""

    payload = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    if payload.get("model_type") != "bc":
        raise ValueError(f"unsupported checkpoint model_type: {payload.get('model_type')}")
    policy = BCPolicy(**payload["model_config"])
    policy.load_state_dict(payload["model_state"])
    policy.eval()
    return policy


__all__ = ["load_bc_policy"]
