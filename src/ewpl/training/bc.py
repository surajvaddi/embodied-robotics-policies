"""Behavior cloning training loop."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Union

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ewpl.data.torch_dataset import CanonicalStepDataset
from ewpl.models.policies.bc import BCPolicy


@dataclass
class BCTrainingConfig:
    dataset_root: Union[str, Path]
    output_dir: Union[str, Path] = "artifacts/training/bc_smoke"
    max_steps: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    seed: int = 0
    image_size: int = 32
    language_length: int = 16
    vocab_size: int = 4096
    hidden_dim: int = 64
    action_convention: str = "delta_ee_pose_gripper"


@dataclass
class TrainingResult:
    checkpoint_path: Path
    metrics_path: Path
    losses: List[float]


def train_bc(config: BCTrainingConfig) -> TrainingResult:
    """Train a small BC policy on canonical per-step samples."""

    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    torch.manual_seed(config.seed)
    dataset = CanonicalStepDataset(
        config.dataset_root,
        image_size=config.image_size,
        language_length=config.language_length,
        vocab_size=config.vocab_size,
    )
    if len(dataset) == 0:
        raise ValueError(f"dataset has no steps: {config.dataset_root}")

    first = dataset[0]
    proprio_dim = int(first["proprio"].numel())
    action_dim = int(first["action"].numel())
    model = BCPolicy(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        image_size=config.image_size,
        language_length=config.language_length,
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        action_convention=config.action_convention,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    iterator = iter(loader)

    losses: List[float] = []
    model.train()
    for _step in range(1, config.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        prediction = model(batch)
        loss = F.mse_loss(prediction, batch["action"].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    checkpoint_path = output_dir / "latest.pt"
    _write_metrics(metrics_path, losses)
    _save_checkpoint(checkpoint_path, model, config, proprio_dim, action_dim, losses)
    return TrainingResult(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        losses=losses,
    )


def _write_metrics(path: Path, losses: List[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss"])
        writer.writeheader()
        for step, loss in enumerate(losses, start=1):
            writer.writerow({"step": step, "loss": f"{loss:.8f}"})


def _save_checkpoint(
    path: Path,
    model: BCPolicy,
    config: BCTrainingConfig,
    proprio_dim: int,
    action_dim: int,
    losses: List[float],
) -> None:
    torch.save(
        {
            "model_type": "bc",
            "model_state": model.state_dict(),
            "model_config": {
                "proprio_dim": proprio_dim,
                "action_dim": action_dim,
                "image_size": config.image_size,
                "language_length": config.language_length,
                "vocab_size": config.vocab_size,
                "hidden_dim": config.hidden_dim,
                "action_convention": config.action_convention,
            },
            "training_config": _serializable_config(config),
            "losses": losses,
        },
        path,
    )


def _serializable_config(config: BCTrainingConfig) -> dict:
    payload = asdict(config)
    payload["dataset_root"] = str(payload["dataset_root"])
    payload["output_dir"] = str(payload["output_dir"])
    return payload


__all__ = ["BCTrainingConfig", "TrainingResult", "train_bc"]
