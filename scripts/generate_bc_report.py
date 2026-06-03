#!/usr/bin/env python
"""Generate behavior cloning smoke report artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ewpl.reports.training import generate_bc_smoke_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="reports/bc_smoke_report.tex")
    parser.add_argument("--figures_dir", default="reports/figures")
    parser.add_argument("--title", default="Behavior Cloning Smoke Report")
    args = parser.parse_args()

    path = generate_bc_smoke_report(
        metrics_csv=args.metrics,
        checkpoint_path=args.checkpoint,
        out=args.out,
        figures_dir=args.figures_dir,
        title=args.title,
    )
    print(f"Wrote report to {path}")


if __name__ == "__main__":
    main()
