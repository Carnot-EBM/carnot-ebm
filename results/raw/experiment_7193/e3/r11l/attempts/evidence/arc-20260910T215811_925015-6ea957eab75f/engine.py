import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    if action == 6 and data is not None:
        # Click observed to flip the top-left indicator cell to the background color 5.
        if g[0, 0] == 0:
            g[0, 0] = 5
    return g

def is_level_complete(grid):
    return False
