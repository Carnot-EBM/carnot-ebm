"""Codex example-conditioned cn04 world model for Exp 4434."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _point(data: Any) -> tuple[int, int] | None:
    if not isinstance(data, Mapping):
        return None
    try:
        return int(data["x"]), int(data["y"])
    except (KeyError, TypeError, ValueError):
        return None


def engine(grid: Any, action: int, data: Any = None) -> np.ndarray:
    """Predict cn04's bounded click/region-toggle transitions."""

    out = np.array(grid, copy=True)
    if out.ndim != 2:
        return out

    height, width = out.shape
    if action == 6:
        point = _point(data)
        if point is None:
            return out
        x, y = point
        if 0 <= y < height and 0 <= x < width and int(out[y, x]) == 4:
            out[y, x] = 0
        return out

    if action == 1:
        for row in range(8, min(14, height)):
            for col in range(11, min(26, width)):
                if int(out[row, col]) == 10:
                    out[row, col] = 0
                elif int(out[row, col]) == 0 and 11 <= row <= 13 and 14 <= col <= 22:
                    out[row, col] = 10
        return out

    if action == 5:
        for row in range(8, min(16, height)):
            for col in range(11, min(29, width)):
                if 11 <= col <= 16:
                    if int(out[row, col]) in (0, 10):
                        out[row, col] = 8
                elif 20 <= col <= 28:
                    if int(out[row, col]) == 0:
                        out[row, col] = 10
                    elif int(out[row, col]) == 10:
                        out[row, col] = 0
        return out

    return out


def is_level_complete(grid: Any) -> bool:
    arr = np.asarray(grid)
    return bool(arr.ndim == 2 and not np.any(arr == 4))
