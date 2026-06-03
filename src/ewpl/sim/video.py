"""Frame writing helpers for rollout artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


def write_frame(rgb, path: Union[str, Path]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(out)
    return out

