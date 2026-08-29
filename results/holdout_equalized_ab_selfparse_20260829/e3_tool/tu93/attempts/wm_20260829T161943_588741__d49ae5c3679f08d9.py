import numpy as np

def _find_obj(grid):
    h, w = grid.shape
    ys, xs = np.where(grid == 4)
    if len(ys) == 0:
        return None
    r, c = int(ys[0]), int(xs[0])
    for r0 in (r - 2, r - 1, r):
        for c0 in (c - 2, c - 1, c):
            if r0 < 0 or c0 < 0 or r0 + 2 >= h or c0 + 2 >= w:
                continue
            ok = True
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    if grid[rr, cc] not in (4, 9):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return (r0, c0)
    return None

def engine(grid, action, data):
    g = grid.astype(int).copy()
    h, w = g.shape
    obj = _find_obj(g)
    if obj is None:
        return g
    r0, c0 = obj
    dr, dc = 0, 0
    mark = None
    if action == 1:
        dr, dc, mark = -6, 0, (0, 1)
    elif action == 2:
        dr, dc, mark = 6, 0, (2, 1)
    elif action == 3:
        dr, dc, mark = 0, -6, (1, 0)
    elif action == 4:
        dr, dc, mark = 0, 6, (1, 2)
    else:
        return g
    nr0, nc0 = r0 + dr, c0 + dc
    if nr0 < 0 or nc0 < 0 or nr0 + 2 >= h or nc0 + 2 >= w:
        return g
    for rr in range(r0, r0 + 3):
        for cc in range(c0, c0 + 3):
            g[rr, cc] = 0
    for rr in range(nr0, nr0 + 3):
        for cc in range(nc0, nc0 + 3):
            g[rr, cc] = 9
    g[nr0 + mark[0], nc0 + mark[1]] = 4
    row = h - 1
    for cc in range(w - 1, -1, -1):
        if g[row, cc] == 6:
            g[row, cc] = 0
            break
    return g

def is_level_complete(grid):
    return False
