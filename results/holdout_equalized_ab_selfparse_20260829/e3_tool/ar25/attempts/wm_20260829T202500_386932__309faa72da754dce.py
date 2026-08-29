import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    H, W = g.shape
    if action == 3:
        g[:, 3:] = g[:, :-3]
        g[:, :3] = 9
    elif action == 2:
        g[3:, :] = g[:-3, :]
        g[:3, :] = 9
    return g

def is_level_complete(grid):
    return False
