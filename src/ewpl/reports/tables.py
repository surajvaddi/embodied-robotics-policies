"""Table generation for dataset audit reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


def dataset_summary_rows(stats: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"metric": "episodes", "value": str(stats.get("episodes", 0))},
        {"metric": "total_steps", "value": str(stats.get("total_steps", 0))},
        {
            "metric": "steps_per_episode_mean",
            "value": f"{stats.get('steps_per_episode_summary', {}).get('mean', 0.0):.3f}",
        },
        {
            "metric": "action_l2_mean",
            "value": f"{stats.get('action_l2_summary', {}).get('mean', 0.0):.3f}",
        },
        {
            "metric": "language_length_mean",
            "value": f"{stats.get('language_length_summary', {}).get('mean', 0.0):.3f}",
        },
    ]


def write_csv_table(rows: Iterable[Dict[str, str]], out: Union[str, Path]) -> Path:
    rows = list(rows)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["metric", "value"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_latex_table(rows: Iterable[Dict[str, str]], out: Union[str, Path]) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{ll}",
        "\\hline",
        "Metric & Value \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(f"{_escape(row['metric'])} & {_escape(row['value'])} \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_dataset_tables(stats: Dict[str, Any], out_dir: Union[str, Path]) -> Dict[str, str]:
    out = Path(out_dir)
    rows = dataset_summary_rows(stats)
    paths = {
        "summary_csv": write_csv_table(rows, out / "dataset_summary.csv"),
        "summary_tex": write_latex_table(rows, out / "dataset_summary.tex"),
    }
    return {key: str(path) for key, path in paths.items()}


def _escape(value: str) -> str:
    return value.replace("_", "\\_")

