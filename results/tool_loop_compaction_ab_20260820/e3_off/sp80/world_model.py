import numpy as np

def _move_color_block(grid, color, dr, dc):
    """Shift all cells of `color` by (dr, dc); vacated cells become 12 (background)."""
    g = grid.copy()
    H, W = g.shape
    mask = (grid == color)
    if not mask.any():
        return g
    ys, xs = np.where(mask)
    # destination
    ny = ys + dr
    nx = xs + dc
    valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
    # clear source
    g[ys, xs] = 12
    # place dest (only valid)
    g[ny[valid], nx[valid]] = color
    return g

def _erode_top_row(grid, n):
    """Erode the topmost row of color 14 by n cells from the right -> 0."""
    g = grid.copy()
    H, W = g.shape
    # find top row that is (mostly) 14
    for r in range(H):
        if (g[r] == 14).sum() > 0:
            row = g[r]
            # rightmost 14 cells
            idx = np.where(row == 14)[0]
            if len(idx) == 0:
                break
            take = idx[-n:] if len(idx) >= n else idx
            g[r, take] = 0
            break
    return g

def engine(grid, action, data):
    g = grid.copy()
    if action == 4:      # right
        g = _move_color_block(g, 9, 0, 4)
        g = _erode_top_row(g, 2)
    elif action == 3:    # left
        g = _move_color_block(g, 9, 0, -4)
        g = _erode_top_row(g, 2)
    elif action == 1:    # up
        g = _move_color_block(g, 9, -4, 0)
        g = _erode_top_row(g, 2)
    elif action == 2:    # down
        g = _move_color_block(g, 9, 4, 0)
        g = _erode_top_row(g, 2)
    return g

def is_level_complete(grid):
    # guess: complete when the 9 block has reached the right edge region
    return False
