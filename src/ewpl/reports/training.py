"""Training report artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Union

from PIL import Image, ImageDraw

from ewpl.reports.latex import write_tex


def read_training_metrics(path: Union[str, Path]) -> Dict[str, object]:
    """Read trainer metrics CSV and compute compact loss statistics."""

    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({"step": int(row["step"]), "loss": float(row["loss"])})
    if not rows:
        raise ValueError(f"training metrics are empty: {path}")
    losses = [float(row["loss"]) for row in rows]
    return {
        "steps": len(rows),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "loss_delta": losses[-1] - losses[0],
        "losses": losses,
    }


def plot_training_loss(metrics: Dict[str, object], out: Union[str, Path]) -> Path:
    """Write a small PNG loss curve for smoke training runs."""

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    losses = list(metrics["losses"])
    width, height = 720, 420
    image = Image.new("RGB", (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), "Behavior Cloning Training Loss", fill=(20, 20, 20))

    left, top, right, bottom = 64, 64, width - 32, height - 56
    draw.line((left, bottom, right, bottom), fill=(80, 80, 80), width=1)
    draw.line((left, top, left, bottom), fill=(80, 80, 80), width=1)

    min_loss = min(losses)
    max_loss = max(losses)
    span = max(max_loss - min_loss, 1e-8)
    points = []
    for idx, loss in enumerate(losses):
        x = left if len(losses) == 1 else left + int(idx / (len(losses) - 1) * (right - left))
        y = bottom - int((loss - min_loss) / span * (bottom - top))
        points.append((x, y))

    if len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(68, 120, 190))
    else:
        draw.line(points, fill=(68, 120, 190), width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(68, 120, 190))

    draw.text((left, bottom + 12), "step 1", fill=(40, 40, 40))
    draw.text((right - 64, bottom + 12), f"step {len(losses)}", fill=(40, 40, 40))
    draw.text((left + 8, top + 8), f"min {min_loss:.4f}", fill=(40, 40, 40))
    draw.text((left + 8, top + 28), f"final {losses[-1]:.4f}", fill=(40, 40, 40))
    image.save(path)
    return path


def render_bc_smoke_report_tex(
    *,
    title: str,
    metrics: Dict[str, object],
    metrics_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    plot_path: Union[str, Path],
) -> str:
    return "\n".join(
        [
            "\\documentclass{article}",
            "\\usepackage[margin=1in]{geometry}",
            "\\usepackage{graphicx}",
            "\\title{" + _escape(title) + "}",
            "\\date{Generated from EWPL training artifacts}",
            "\\begin{document}",
            "\\maketitle",
            "\\section{Smoke Training Summary}",
            f"Steps: {metrics['steps']}\\\\",
            f"Initial loss: {float(metrics['initial_loss']):.6f}\\\\",
            f"Final loss: {float(metrics['final_loss']):.6f}\\\\",
            f"Minimum loss: {float(metrics['min_loss']):.6f}\\\\",
            f"Loss delta: {float(metrics['loss_delta']):.6f}\\\\",
            "Metrics CSV: \\detokenize{" + str(metrics_path) + "}\\\\",
            "Checkpoint: \\detokenize{" + str(checkpoint_path) + "}",
            "\\section{Loss Curve}",
            "\\begin{figure}[h]",
            "\\centering",
            "\\includegraphics[width=0.85\\linewidth]{\\detokenize{" + str(plot_path) + "}}",
            "\\caption{Behavior cloning smoke training loss}",
            "\\end{figure}",
            "\\end{document}",
            "",
        ]
    )


def generate_bc_smoke_report(
    *,
    metrics_csv: Union[str, Path],
    checkpoint_path: Union[str, Path],
    out: Union[str, Path],
    figures_dir: Union[str, Path] = "reports/figures",
    title: str = "Behavior Cloning Smoke Report",
) -> Path:
    metrics = read_training_metrics(metrics_csv)
    plot_path = plot_training_loss(metrics, Path(figures_dir) / "bc_training_loss.png")
    tex = render_bc_smoke_report_tex(
        title=title,
        metrics=metrics,
        metrics_path=metrics_csv,
        checkpoint_path=checkpoint_path,
        plot_path=plot_path,
    )
    return write_tex(tex, out)


def _escape(value: str) -> str:
    return value.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&")


__all__ = [
    "generate_bc_smoke_report",
    "plot_training_loss",
    "read_training_metrics",
    "render_bc_smoke_report_tex",
]
