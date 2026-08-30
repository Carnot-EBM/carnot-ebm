import numpy as np

def _player_bbox(g):
    mask = (g == 12)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r0 = int(np.argmax(rows)); c0 = int(np.argmax(cols))
    return r0, c0, r0 + 4, c0 + 4

def _move_player(g, dr, dc):
    bb = _player_bbox(g)
    if bb is None:
        return g
    r0, c0, r1, c1 = bb
    nr0, nc0, nr1, nc1 = r0 + dr, c0 + dc, r1 + dr, c1 + dc
    H, W = g.shape
    if nr0 < 0 or nc0 < 0 or nr1 >= H or nc1 >= W:
        return g
    patch = g[r0:r1 + 1, c0:c1 + 1].copy()
    g2 = g.copy()
    g2[r0:r1 + 1, c0:c1 + 1] = 3
    g2[nr0:nr1 + 1, nc0:nc1 + 1] = patch
    return g2

def _advance_bar(g):
    # find the 11-run on the bottom bar rows (rows with a long 11 run)
    H, W = g.shape
    for r in range(H - 1, -1, -1):
        row = g[r]
        # find runs of 11
        idx = np.where(row == 11)[0]
        if len(idx) >= 5:
            left = int(idx.min())
            # find the 3 immediately to the left of the 11 run
            if left > 0 and row[left - 1] == 3:
                g[r, left] = 3
                if r + 1 < H and g[r + 1, left] == 11:
                    g[r + 1, left] = 3
                return g
    return g

def engine(grid, action, data):
    g = np.array(grid, dtype=int)
    if action == 1:
        g = _move_player(g, -5, 0)
    elif action == 2:
        g = _move_player(g, 5, 0)
    elif action == 3:
        g = _move_player(g, 0, -5)
    elif action == 4:
        g = _move_player(g, 0, 5)
    g = _advance_bar(g)
    return g

def is_level_complete(grid):
    return False
