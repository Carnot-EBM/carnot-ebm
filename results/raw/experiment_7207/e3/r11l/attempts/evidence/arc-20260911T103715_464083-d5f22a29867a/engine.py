import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    if action != 6 or data is None:
        return g
    x = int(data.get('x', 0))
    y = int(data.get('y', 0))
    r, c = y, x
    H, W = g.shape
    if not (0 <= r < H and 0 <= c < W):
        return g
    # click on a 6 (green core) -> move it down-right by (6,5), leaving a 15 core
    if g[r, c] == 6:
        nr, nc = r + 6, c + 5
        if 0 <= nr < H and 0 <= nc < W:
            g[nr, nc] = 6
        g[r, c] = 15
    return g

def is_level_complete(grid):
    return False
