import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    if action == 6 and data is not None:
        # rough: click triggers a change at top-left corner
        if g[0, 0] == 0:
            g[0, 0] = 5
    return g

def is_level_complete(grid):
    return False
