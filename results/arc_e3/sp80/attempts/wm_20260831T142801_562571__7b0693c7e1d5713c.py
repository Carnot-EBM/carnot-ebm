import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    H, W = g.shape
    # moving block: color 9 if present else color 11
    if 9 in g:
        mask = (g == 9)
    else:
        mask = (g == 11)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) and len(cols):
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        h = r1 - r0 + 1
        step = h
        dr = dc = 0
        if action == 1:
            dr = -step
        elif action == 2:
            dr = step
        elif action == 3:
            dc = -step
        elif action == 4:
            dc = step
        nr0, nr1 = r0 + dr, r1 + dr
        nc0, nc1 = c0 + dc, c1 + dc
        if 0 <= nr0 and nr1 < H and 0 <= nc0 and nc1 < W:
            block = g[r0:r1 + 1, c0:c1 + 1]
            g[r0:r1 + 1, c0:c1 + 1] = 12
            g[nr0:nr1 + 1, nc0:nc1 + 1] = block
    # top bar (color 14) erodes from the right: 3 for a click (action 6), else 2
    if 14 in g:
        bar = np.where(g[0] == 14)[0]
        if len(bar):
            right = int(bar.max())
            n = 3 if action == 6 else 2
            g[0, max(0, right - n + 1):right + 1] = 0
    return g

def is_level_complete(grid):
    g = np.asarray(grid)
    return bool((g == 14).sum() == 0)
