import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64, copy=True)
    if action != 6 or data is None:
        return g
    x = int(data.get('x', 0)); y = int(data.get('y', 0))
    H, W = g.shape
    slots = []
    for r in range(H - 3):
        for c in range(W - 3):
            blk = g[r:r+4, c:c+4]
            if not np.all(blk == blk[0, 0]):
                continue
            v = int(blk[0, 0])
            if v == 4:
                continue
            if r > 0 and c > 0 and r + 4 < H and c + 4 < W:
                if not (np.all(g[r-1, c:c+4] == 4) and np.all(g[r+4, c:c+4] == 4)
                        and np.all(g[r:r+4, c-1] == 4) and np.all(g[r:r+4, c+4] == 4)):
                    continue
            slots.append((r, c, v))
    if not slots:
        return g
    rows = sorted(set(r for r, c, v in slots))
    cols = sorted(set(c for r, c, v in slots))
    val = {}
    for r, c, v in slots:
        val[(r, c)] = v
    rc = (rows[0] + rows[-1]) / 2.0
    cc = (cols[0] + cols[-1]) / 2.0
    dcol = -1 if x < cc else (1 if x > cc else 0)
    drow = -1 if y < rc else (1 if y > rc else 0)
    newval = {}
    for (r, c), v in val.items():
        nr = r + drow
        nc = c + dcol
        if nr in rows and nc in cols:
            newval[(nr, nc)] = v
    for (r, c), v in newval.items():
        g[r:r+4, c:c+4] = v
    return g

def is_level_complete(grid):
    return False
