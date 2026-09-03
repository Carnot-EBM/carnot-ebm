import numpy as np


def engine(grid, action, data):
    g = np.array(grid, copy=True)

    if g.ndim != 2 or g.shape[0] == 0 or g.shape[1] == 0:
        return g

    # Observed click behavior: consume the first remaining cell of the
    # vertical progress indicator in column 0, filling it with color 5.
    if action == 6:
        rows = np.flatnonzero(g[:, 0] == 0)
        if rows.size > 0:
            g[int(rows[0]), 0] = 5

    return g


def is_level_complete(grid):
    g = np.asarray(grid)

    if g.ndim != 2 or g.shape[0] == 0 or g.shape[1] == 0:
        return False

    # The level is complete when the vertical zero-colored progress bar
    # in the first column has been fully consumed/filled with color 5.
    return bool(np.all(g[:, 0] == 5))