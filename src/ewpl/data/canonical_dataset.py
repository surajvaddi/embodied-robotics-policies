"""Dataset accessors for canonical robotics episodes."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from ewpl.data.schemas import Episode
from ewpl.data.storage import CanonicalEpisodeStorage


class CanonicalDataset:
    """Small filesystem-backed canonical dataset."""

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        self.storage = CanonicalEpisodeStorage(self.root)
        self.episode_ids = self.storage.list_episode_ids()

    def __len__(self) -> int:
        return len(self.episode_ids)

    def __getitem__(self, index: int) -> Episode:
        return self.storage.load_episode(self.episode_ids[index])

    def episodes(self) -> List[Episode]:
        return [self[index] for index in range(len(self))]
