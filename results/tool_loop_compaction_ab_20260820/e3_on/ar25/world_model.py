import numpy as np

def _move_object(g, color, dr, dc, fill):
    """Move ALL cells of `color` by (dr, dc). Cells vacated become `fill`."""
    H, W = g.shape
    mask = (g == color)
    if not mask.any():
        return
    rs, cs = np.where(mask)
    nr = rs + dr
    nc = cs + dc
    if (nr < 0).any() or (nr >= H).any() or (nc < 0).any() or (nc >= W).any():
        return  # blocked: object does not move
    g[rs, cs] = fill
    g[nr, nc] = color

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    H, W = g.shape
    # progress: the 5-colored fill grows one cell down the rightmost column
    col = W - 1
    r = 0
    while r < H and g[r, col] == 5:
        r += 1
    if r < H and g[r, col] != 5:
        g[r, col] = 5
    if action == 3:
        _move_object(g, 5, 0, -3, 9)
        _move_object(g, 4, 0, +3, 9)
    elif action == 2:
        _move_object(g, 5, +3, 0, 9)
        _move_object(g, 4, +3, 0, 9)
    return g

def is_level_complete(grid):
    g = np.asarray(grid)
    # win when the 5-colored player object reaches the bottom row
    return bool((g[-1] == 5).any())