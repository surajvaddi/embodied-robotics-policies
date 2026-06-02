#!/usr/bin/env python
"""Generate report artifacts from canonical datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ewpl.data.canonical_dataset import CanonicalDataset
from ewpl.eval.statistics import compute_dataset_statistics, write_statistics
from ewpl.reports.latex import render_dataset_audit_tex, write_tex
from ewpl.reports.plots import generate_dataset_plots
from ewpl.reports.tables import generate_dataset_tables


def generate_dataset_audit(config_path: Union[str, Path], out: Union[str, Path] = None) -> Path:
    config = _load_simple_yaml(config_path)
    dataset = CanonicalDataset(config["dataset"])
    stats = compute_dataset_statistics(dataset)
    write_statistics(stats, config.get("statistics_out", "artifacts/reports/dataset_audit_statistics.json"))
    plot_paths = generate_dataset_plots(stats, config.get("figures_dir", "reports/figures"))
    table_paths = generate_dataset_tables(stats, config.get("tables_dir", "reports/tables"))
    tex = render_dataset_audit_tex(
        title=config.get("title", "Dataset Audit"),
        stats=stats,
        plot_paths=plot_paths,
        table_paths=table_paths,
        known_limitations=config.get("known_limitations", []),
    )
    return write_tex(tex, out or config.get("out", "reports/dataset_audit.tex"))


def _load_simple_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    current_list_key = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].rstrip()
        if not line:
            continue
        if line.startswith("  - ") and current_list_key:
            payload.setdefault(current_list_key, []).append(line[4:].strip())
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
            payload[key] = value.strip("\"'")
            current_list_key = None
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True, choices=["dataset_audit"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.type == "dataset_audit":
        path = generate_dataset_audit(args.config, out=args.out)
    else:
        raise ValueError(f"unsupported report type: {args.type}")
    print(f"Wrote report to {path}")


if __name__ == "__main__":
    main()
