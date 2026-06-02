"""Train/validation/test split generation for canonical datasets."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Union

from ewpl.data.canonical_dataset import CanonicalDataset
from ewpl.data.schemas import Episode


@dataclass
class SplitConfig:
    seed: int = 0
    split_type: str = "iid"
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    task_holdout_fraction: float = 0.25
    source_transfer_train_sources: List[str] = None
    source_transfer_test_sources: List[str] = None

    def __post_init__(self) -> None:
        if self.source_transfer_train_sources is None:
            self.source_transfer_train_sources = ["lerobot", "openx"]
        if self.source_transfer_test_sources is None:
            self.source_transfer_test_sources = ["libero", "robocasa"]


def load_split_config(path: Union[str, Path]) -> SplitConfig:
    payload = _load_simple_yaml(path)
    return SplitConfig(
        seed=int(payload.get("seed", 0)),
        split_type=str(payload.get("split_type", "iid")),
        train_fraction=float(payload.get("train_fraction", 0.7)),
        val_fraction=float(payload.get("val_fraction", 0.15)),
        test_fraction=float(payload.get("test_fraction", 0.15)),
        task_holdout_fraction=float(payload.get("task_holdout_fraction", 0.25)),
        source_transfer_train_sources=list(payload.get("source_transfer_train_sources") or []),
        source_transfer_test_sources=list(payload.get("source_transfer_test_sources") or []),
    )


def make_split_manifest(
    episodes: Sequence[Episode],
    *,
    config: SplitConfig,
    split_type: str = None,
) -> Dict[str, Any]:
    selected_type = split_type or config.split_type
    if selected_type == "iid":
        splits = make_iid_split(episodes, config)
    elif selected_type == "task_holdout":
        splits = make_task_holdout_split(episodes, config)
    elif selected_type == "source_transfer":
        splits = make_source_transfer_split(episodes, config)
    else:
        raise ValueError(f"unsupported split_type: {selected_type}")

    assert_no_leakage(splits)
    return {
        "split_type": selected_type,
        "seed": config.seed,
        "counts": {name: len(ids) for name, ids in splits.items()},
        "splits": splits,
    }


def make_iid_split(episodes: Sequence[Episode], config: SplitConfig) -> Dict[str, List[str]]:
    ids = [episode.episode_id for episode in episodes]
    rng = random.Random(config.seed)
    rng.shuffle(ids)

    total = len(ids)
    train_n = int(total * config.train_fraction)
    val_n = int(total * config.val_fraction)
    if total > 0 and train_n == 0:
        train_n = 1
    if total >= 3 and val_n == 0:
        val_n = 1
    train = ids[:train_n]
    val = ids[train_n : train_n + val_n]
    test = ids[train_n + val_n :]
    return {"train": train, "val": val, "test": test}


def make_task_holdout_split(
    episodes: Sequence[Episode], config: SplitConfig
) -> Dict[str, List[str]]:
    by_task: Dict[str, List[str]] = {}
    for episode in episodes:
        by_task.setdefault(episode.task_id, []).append(episode.episode_id)

    tasks = sorted(by_task)
    rng = random.Random(config.seed)
    rng.shuffle(tasks)
    holdout_n = max(1, int(len(tasks) * config.task_holdout_fraction)) if tasks else 0
    test_tasks = set(tasks[:holdout_n])
    train_val_ids = [episode.episode_id for episode in episodes if episode.task_id not in test_tasks]
    test_ids = [episode.episode_id for episode in episodes if episode.task_id in test_tasks]

    rng.shuffle(train_val_ids)
    val_n = int(len(train_val_ids) * config.val_fraction)
    if len(train_val_ids) >= 2 and val_n == 0:
        val_n = 1
    return {"train": train_val_ids[val_n:], "val": train_val_ids[:val_n], "test": test_ids}


def make_source_transfer_split(
    episodes: Sequence[Episode], config: SplitConfig
) -> Dict[str, List[str]]:
    train_sources = set(config.source_transfer_train_sources)
    test_sources = set(config.source_transfer_test_sources)
    train = [episode.episode_id for episode in episodes if episode.source in train_sources]
    test = [episode.episode_id for episode in episodes if episode.source in test_sources]
    return {"train": train, "val": [], "test": test}


def assert_no_leakage(splits: Dict[str, Iterable[str]]) -> None:
    seen: Dict[str, str] = {}
    for split_name, ids in splits.items():
        for episode_id in ids:
            if episode_id in seen:
                raise ValueError(
                    f"episode {episode_id!r} appears in both {seen[episode_id]} and {split_name}"
                )
            seen[episode_id] = split_name


def write_split_manifest(manifest: Dict[str, Any], out: Union[str, Path]) -> Path:
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "splits.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_simple_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    current_list_key = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].rstrip()
        if not line:
            continue
        if line.startswith("  - ") and current_list_key:
            payload.setdefault(current_list_key, []).append(_parse_scalar(line[4:].strip()))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if value == "":
            payload[key] = []
            current_list_key = key
        else:
            payload[key] = _parse_scalar(value)
            current_list_key = None
    return payload


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split_type", default=None)
    args = parser.parse_args()

    dataset = CanonicalDataset(args.dataset)
    config = load_split_config(args.config)
    manifest = make_split_manifest(dataset.episodes(), config=config, split_type=args.split_type)
    path = write_split_manifest(manifest, args.out)
    print(f"Wrote split manifest to {path}")
    print(f"Split type: {manifest['split_type']}")
    print(f"Counts: {manifest['counts']}")


if __name__ == "__main__":
    main()

