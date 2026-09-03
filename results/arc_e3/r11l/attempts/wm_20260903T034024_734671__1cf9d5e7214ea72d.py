import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    if action == 6 and data is not None:
        try:
            x = int(data.get('x', -1))
            y = int(data.get('y', -1))
        except Exception:
            x, y = -1, -1
        if 0 <= y < g.shape[0] and 0 <= x < g.shape[1]:
            v = int(g[y, x])
            # Observed: clicking the core (color 6) of a gem cluster flips the
            # top-left corner cell from 0 to the background color 5.
            if v == 6:
                g[0, 0] = 5
    return g

def is_level_complete(grid):
    g = np.asarray(grid)
    # No win state observed; opening boards must be False.
    return False
