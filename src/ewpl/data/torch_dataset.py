"""Torch dataset adapters for canonical robotics episodes."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - exercised by non-torch environments.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

from ewpl.data.canonical_dataset import CanonicalDataset
from ewpl.data.schemas import Episode
from ewpl.data.tensorization import tensorize_step


class CanonicalStepDataset(Dataset):
    """Per-step Torch view over a filesystem-backed canonical dataset."""

    def __init__(
        self,
        root: Union[str, Path],
        *,
        image_size: int = 32,
        language_length: int = 16,
        vocab_size: int = 4096,
    ) -> None:
        if torch is None:
            raise ImportError("CanonicalStepDataset requires torch to be installed")

        self.dataset = CanonicalDataset(root)
        self.image_size = image_size
        self.language_length = language_length
        self.vocab_size = vocab_size
        self._episodes = self.dataset.episodes()
        self._index = _build_step_index(self._episodes)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> Dict[str, object]:
        episode_index, step_index = self._index[index]
        episode = self._episodes[episode_index]
        step = episode.steps[step_index]
        tensorized = tensorize_step(
            step,
            image_size=self.image_size,
            language_length=self.language_length,
            vocab_size=self.vocab_size,
        )
        return {
            "image": torch.as_tensor(tensorized.image, dtype=torch.float32),
            "proprio": torch.as_tensor(tensorized.proprio, dtype=torch.float32),
            "language_ids": torch.as_tensor(tensorized.language_ids, dtype=torch.long),
            "action": torch.as_tensor(tensorized.action, dtype=torch.float32),
            "episode_id": episode.episode_id,
            "step_index": step_index,
        }


def _build_step_index(episodes: List[Episode]) -> List[Tuple[int, int]]:
    index: List[Tuple[int, int]] = []
    for episode_index, episode in enumerate(episodes):
        for step_index in range(len(episode.steps)):
            index.append((episode_index, step_index))
    return index


__all__ = ["CanonicalStepDataset"]
