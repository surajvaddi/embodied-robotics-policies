"""Repo-root import shim for running src-layout modules before installation."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.1.0"

SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "ewpl"
if SRC_PACKAGE.exists():
    __path__.append(str(SRC_PACKAGE))
