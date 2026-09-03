import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    if action != 6 or not isinstance(data, dict):
        return g
    px, py = int(data.get('x', 0)), int(data.get('y', 0))
    r, c = py, px
    H, W = g.shape
    if not (0 <= r < H and 0 <= c < W):
        return g
    if g[r, c] != 6:
        return g
    # move the 6-pixel one step down-right; blocked by non-5 cells
    nr, nc = r + 1, c + 1
    if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == 5:
        g[nr, nc] = 6
        g[r, c] = 15
    return g

def is_level_complete(grid):
    return False