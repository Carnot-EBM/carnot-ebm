import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    H, W = g.shape
    if action == 6 and data is not None:
        px, py = int(data.get('x', 0)), int(data.get('y', 0))
        c, r = px // 1, py // 1
        if 0 <= r < H and 0 <= c < W:
            g[r, c] = 3
    return g

def is_level_complete(grid):
    return False
