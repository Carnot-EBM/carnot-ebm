import numpy as np

def _find_obj(grid):
    mask = (grid == 9) | (grid == 4)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r0 = int(np.argmax(rows)); r1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols)); c1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return r0, c0, r1, c1

def engine(grid, action, data):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    box = _find_obj(g)
    if box is None:
        return g
    r0, c0, r1, c1 = box
    d = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action, (0, 0))
    nr0, nc0 = r0 + d[0]*6, c0 + d[1]*6
    nr1, nc1 = nr0 + (r1-r0), nc0 + (c1-c0)
    if 0 <= nr0 and nr1 < H and 0 <= nc0 and nc1 < W:
        g[r0:r1+1, c0:c1+1] = 0
        g[nr0:nr1+1, nc0:nc1+1] = 9
        cr = (nr0 + nr1) // 2
        cc = (nc0 + nc1) // 2
        if d == (1, 0):
            g[nr1, cc] = 4
        elif d == (-1, 0):
            g[nr0, cc] = 4
        elif d == (0, 1):
            g[cr, nc1] = 4
        elif d == (0, -1):
            g[cr, nc0] = 4
    # bottom bar erosion: remove from the right end of the 6-run
    if H > 0:
        row = g[H-1]
        i = W - 1
        while i >= 0 and row[i] == 0:
            i -= 1
        if i >= 0 and row[i] == 6:
            # find left end of the contiguous 6-run ending at i
            j = i
            while j >= 0 and row[j] == 6:
                j -= 1
            length = i - j
            n = 2 if (length % 5 == 4) else 1
            for k in range(n):
                if i - k >= 0 and row[i-k] == 6:
                    row[i-k] = 0
    return g

def is_level_complete(grid):
    g = np.asarray(grid)
    return bool(((g == 9) | (g == 4)).sum() == 0)