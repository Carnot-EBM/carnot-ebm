import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    if action == 6 and data is not None:
        px, py = int(data.get('x', 0)), int(data.get('y', 0))
        r, c = py // 1, px // 1
        if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
            g[r, c] = 12
    return g

def is_level_complete(grid):
    return False