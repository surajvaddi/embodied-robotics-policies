"""Lightweight plot generation for dataset audit reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

from PIL import Image, ImageDraw


def plot_bar_counts(
    counts: Dict[str, int],
    out: Union[str, Path],
    *,
    title: str,
    width: int = 720,
    height: int = 420,
) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), title, fill=(20, 20, 20))

    if not counts:
        draw.text((24, 72), "No data", fill=(80, 80, 80))
        image.save(path)
        return path

    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12]
    max_count = max(value for _, value in items)
    bar_left = 210
    bar_top = 64
    bar_height = 22
    gap = 9
    bar_max_width = width - bar_left - 72

    for idx, (label, value) in enumerate(items):
        y = bar_top + idx * (bar_height + gap)
        draw.text((24, y + 3), _trim(label, 28), fill=(30, 30, 30))
        bar_width = int((value / max_count) * bar_max_width) if max_count else 0
        draw.rectangle((bar_left, y, bar_left + bar_width, y + bar_height), fill=(68, 120, 190))
        draw.text((bar_left + bar_width + 8, y + 3), str(value), fill=(30, 30, 30))

    image.save(path)
    return path


def generate_dataset_plots(stats: Dict[str, Any], out_dir: Union[str, Path]) -> Dict[str, str]:
    out = Path(out_dir)
    paths = {
        "task_distribution": plot_bar_counts(
            stats.get("tasks", {}),
            out / "dataset_task_distribution.png",
            title="Episodes Per Task",
        ),
        "source_distribution": plot_bar_counts(
            stats.get("sources", {}),
            out / "dataset_source_distribution.png",
            title="Episodes Per Source",
        ),
        "action_dim_distribution": plot_bar_counts(
            stats.get("action_dims", {}),
            out / "dataset_action_dims.png",
            title="Steps Per Action Dimension",
        ),
    }
    return {key: str(path) for key, path in paths.items()}


def _trim(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "..."

