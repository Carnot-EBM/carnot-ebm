import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    if action == 6 and data is not None:
        # Click: the observed effect is a single cell at the top-left corner
        # flipping from background (0) to the field color (5).
        if g[0, 0] == 0:
            g[0, 0] = 5
    return g

def is_level_complete(grid):
    return False
