import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    if action == 6 and data is not None:
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        # pixel = logical*1
        r, c = y, x
        if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
            g[r, c] = 0
    return g

def is_level_complete(grid):
    return False
