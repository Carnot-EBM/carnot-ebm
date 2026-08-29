import numpy as np

def find_block(g, color):
    H, W = g.shape
    best = None
    bestc = -1
    for r in range(H-4):
        for c in range(W-4):
            sub = g[r:r+5, c:c+5]
            cnt = int(np.sum(sub == color))
            if cnt >= 20 and cnt > bestc:
                bestc = cnt
                best = (r, c)
    return best

def engine(grid, action, data):
    g = grid.copy()
    H, W = g.shape
    p = find_block(g, 9)
    if p is None:
        return g
    r0, c0 = p
    pat = g[r0:r0+5, c0:c0+5].copy()
    dr = dc = 0
    if action == 2:
        dr = 6
    elif action == 4:
        dc = 6
    elif action == 1:
        dr = -6
    elif action == 3:
        dc = -6
    nr0, nc0 = r0+dr, c0+dc
    if 0 <= nr0 and nr0+5 <= H and 0 <= nc0 and nc0+5 <= W:
        g[r0:r0+5, c0:c0+5] = 5
        g[nr0:nr0+5, nc0:nc0+5] = pat
    return g

def is_level_complete(grid):
    return False
