"""Visualization helpers for canonical datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageDraw

from ewpl.data.canonical_dataset import CanonicalDataset


def render_episode_grid(
    dataset_root: Union[str, Path], out: Union[str, Path], *, max_episodes: int = 8
) -> Path:
    dataset = CanonicalDataset(dataset_root)
    if len(dataset) == 0:
        raise ValueError(f"dataset has no episodes: {dataset_root}")

    cells = []
    for episode in dataset.episodes()[:max_episodes]:
        rgb = np.asarray(episode.steps[0].observation.rgb, dtype=np.uint8)
        image = Image.fromarray(rgb).resize((160, 120))
        draw = ImageDraw.Draw(image)
        label = f"{episode.task_id} | success={episode.success}"
        draw.rectangle((0, 0, 160, 24), fill=(0, 0, 0))
        draw.text((4, 6), label[:28], fill=(255, 255, 255))
        cells.append(image)

    cols = min(4, len(cells))
    rows = int(np.ceil(len(cells) / cols))
    canvas = Image.new("RGB", (cols * 160, rows * 120), color=(245, 245, 245))
    for idx, cell in enumerate(cells):
        canvas.paste(cell, ((idx % cols) * 160, (idx // cols) * 120))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_episodes", type=int, default=8)
    args = parser.parse_args()
    path = render_episode_grid(args.dataset, args.out, max_episodes=args.max_episodes)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
