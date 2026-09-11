import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    if action == 6 and data is not None:
        # click: observed to set the top-left cell to the dominant blob color (5)
        g[0, 0] = 5
    return g

def is_level_complete(grid):
    return False
