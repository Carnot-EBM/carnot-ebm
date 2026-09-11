import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    if action != 6 or data is None:
        return g
    H, W = g.shape
    ys, xs = np.where(g == 6)
    if len(ys) == 0:
        return g
    by, bx = int(ys[0]), int(xs[0])
    # ball (6) rolls diagonally down-right through 5/6/15/1/0 until it hits a 2 wall or edge
    ny, nx = by, bx
    while True:
        ry, rx = ny + 1, nx + 1
        if ry >= H or rx >= W:
            break
        v = g[ry, rx]
        if v == 2:
            break
        ny, nx = ry, rx
    if (ny, nx) == (by, bx):
        return g
    g[by, bx] = 15
    g[ny, nx] = 6
    return g

def is_level_complete(grid):
    return False
