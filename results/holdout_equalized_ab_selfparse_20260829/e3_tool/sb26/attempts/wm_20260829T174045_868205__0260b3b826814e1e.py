import numpy as np

def _find_2x2(g, color):
    H, W = g.shape
    out = []
    for r in range(H - 1):
        for c in range(W - 1):
            if g[r, c] == color and g[r, c + 1] == color and g[r + 1, c] == color and g[r + 1, c + 1] == color:
                out.append((r, c))
    return out

def _find_4x4(g, color):
    H, W = g.shape
    out = []
    for r in range(H - 3):
        for c in range(W - 3):
            if (g[r:r + 4, c:c + 4] == color).all():
                out.append((r, c))
    return out

def _nearest(cands, x, y):
    best = None; bestd = 1e18
    for (r, c) in cands:
        d = (r + 1 - y) ** 2 + (c + 1 - x) ** 2
        if d < bestd:
            bestd = d; best = (r, c)
    return best, bestd

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    if action != 6 or data is None:
        return g
    x = int(data.get('x', 0)); y = int(data.get('y', 0))
    H, W = g.shape
    if not (0 <= y < H and 0 <= x < W):
        return g
    dots = _find_2x2(g, 2)
    blocks = _find_4x4(g, 9)
    db, dbd = _nearest(blocks, x, y)
    dd, ddd = _nearest(dots, x, y)
    if db is not None and (dd is None or dbd <= ddd):
        # click on a 4x4 block: clear the 6x6 region around it
        r0, c0 = db
        for r in range(r0 - 1, r0 + 5):
            for c in range(c0 - 1, c0 + 5):
                if 0 <= r < H and 0 <= c < W:
                    g[r, c] = 0
        return g
    if dd is not None:
        # click on a 2x2 dot: place a 4x4 block of color 9 centered on it
        r0, c0 = dd
        for r in range(r0 - 1, r0 + 3):
            for c in range(c0 - 1, c0 + 3):
                if 0 <= r < H and 0 <= c < W:
                    g[r, c] = 9
        return g
    return g

def is_level_complete(grid):
    return False
