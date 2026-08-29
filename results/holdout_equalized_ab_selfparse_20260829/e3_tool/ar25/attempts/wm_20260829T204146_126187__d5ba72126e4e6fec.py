import numpy as np

def _move(g, dr, dc):
    H, W = g.shape
    new = g.copy()
    for r in range(H):
        for c in range(W):
            if g[r, c] == 5:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == 9:
                    new[nr, nc] = 5
                    new[r, c] = 9
    return new

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    if action == 1:
        g = _move(g, -1, 0)
    elif action == 2:
        g = _move(g, 1, 0)
    elif action == 3:
        g = _move(g, 0, -1)
    elif action == 4:
        g = _move(g, 0, 1)
    return g

def is_level_complete(grid):
    return False