"""LaTeX report generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Union


def render_dataset_audit_tex(
    *,
    title: str,
    stats: Dict[str, Any],
    plot_paths: Dict[str, str],
    table_paths: Dict[str, str],
    known_limitations: Sequence[str],
) -> str:
    limitations = "\n".join(f"\\item {_escape(item)}" for item in known_limitations)
    return "\n".join(
        [
            "\\documentclass{article}",
            "\\usepackage[margin=1in]{geometry}",
            "\\usepackage{graphicx}",
            "\\usepackage{booktabs}",
            "\\title{" + _escape(title) + "}",
            "\\date{Generated from canonical EWPL artifacts}",
            "\\begin{document}",
            "\\maketitle",
            "\\section{Dataset Summary}",
            f"Episodes: {stats.get('episodes', 0)}\\\\",
            f"Total steps: {stats.get('total_steps', 0)}\\\\",
            f"Sources: {_escape(str(stats.get('sources', {})))}\\\\",
            f"Tasks: {_escape(str(stats.get('tasks', {})))}",
            "\\section{Summary Table}",
            "\\input{" + _escape(table_paths["summary_tex"]) + "}",
            "\\section{Distributions}",
            _figure(plot_paths["source_distribution"], "Episodes per source"),
            _figure(plot_paths["task_distribution"], "Episodes per task"),
            _figure(plot_paths["action_dim_distribution"], "Steps per action dimension"),
            "\\section{Known Limitations}",
            "\\begin{itemize}",
            limitations,
            "\\end{itemize}",
            "\\end{document}",
            "",
        ]
    )


def write_tex(tex: str, out: Union[str, Path]) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8")
    return path


def _figure(path: str, caption: str) -> str:
    return "\n".join(
        [
            "\\begin{figure}[h]",
            "\\centering",
            "\\includegraphics[width=0.85\\linewidth]{" + _escape(path) + "}",
            "\\caption{" + _escape(caption) + "}",
            "\\end{figure}",
        ]
    )


def _escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )

