import numpy as np


def _click_rc(data):
    """Convert ACTION6 click data to logical (row, col)."""
    x = y = None
    if isinstance(data, dict):
        x = data.get("x", None)
        y = data.get("y", None)
    elif isinstance(data, (tuple, list)) and len(data) >= 2:
        x, y = data[0], data[1]

    try:
        c = int(x)
        r = int(y)
    except Exception:
        return None
    return r, c


def engine(grid, action, data):
    # Pure deterministic prediction.
    g = grid.copy() if isinstance(grid, np.ndarray) else np.array(grid, copy=True)

    try:
        act = int(action)
    except Exception:
        return g

    # In the observed transition, only a mouse click on / near a yellow key
    # (color 6) advanced the left-edge progress marker by one cell.
    if act != 6:
        return g

    rc = _click_rc(data)
    if rc is None:
        return g

    r, c = rc
    h, w = g.shape[:2]
    if not (0 <= r < h and 0 <= c < w):
        return g

    # Treat clicks anywhere on the small key object as activating it.
    r0, r1 = max(0, r - 2), min(h - 1, r + 2)
    c0, c1 = max(0, c - 2), min(w - 1, c + 2)
    if not np.any(g[r0:r1 + 1, c0:c1 + 1] == 6):
        return g

    # Advance the first still-empty cell in the leftmost column to color 5.
    empty_rows = np.flatnonzero(g[:, 0] == 0)
    if len(empty_rows) == 0:
        return g

    g[int(empty_rows[0]), 0] = 5
    return g


def is_level_complete(grid):
    a = np.asarray(grid)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        return False

    # The observed progress marker fills the left edge with color 5.
    # A completed level is taken to be one where that edge is fully filled.
    return bool(np.all(a[:, 0] == 5))