import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    # action 1 = up, 3 = left (hypothesis); step size 5
    if action == 1:
        dr, dc = -5, 0
    elif action == 3:
        dr, dc = 0, -5
    else:
        return g
    # find the moving block: 5x5 region whose top 2 rows are color 12 and bottom 3 rows color 9
    best = None
    for r in range(H - 4):
        for c in range(W - 4):
            sub = g[r:r+5, c:c+5]
            if (sub[0:2] == 12).all() and (sub[2:5] == 9).all():
                best = (r, c)
                break
        if best is not None:
            break
    if best is None:
        return g
    r0, c0 = best
    r1, c1 = r0 + dr, c0 + dc
    if r1 < 0 or c1 < 0 or r1 + 5 > H or c1 + 5 > W:
        return g
    # check destination is clear (background 3 or 4)
    dest = g[r1:r1+5, c1:c1+5]
    if not np.all((dest == 3) | (dest == 4)):
        return g
    g[r0:r0+5, c0:c0+5] = 3
    g[r1:r1+2, c1:c1+5] = 12
    g[r1+2:r1+5, c1:c1+5] = 9
    # the 11 strip at bottom grows by 1 cell of 3 from its left edge each step
    for r in range(H):
        for c in range(1, W):
            if g[r, c] == 11 and g[r, c-1] == 3:
                g[r, c] = 3
                break
    return g

def is_level_complete(grid):
    return False