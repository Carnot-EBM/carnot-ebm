import numpy as np

def _fill_bar(g):
    # progress bar on bottom row: fills 2 cells per click, right-to-left
    row = g[-1]
    n11 = int(np.sum(row == 11))
    n12 = int(np.sum(row == 12))
    if n12 < 2:
        return
    if n11 == 0:
        g[-1, -2:] = 11
    else:
        idx = np.where(row == 11)[0]
        left = int(idx.min())
        if left >= 2:
            g[-1, left - 2:left] = 11

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64, copy=True)
    if action == 6 and data is not None:
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        H, W = g.shape
        if 0 <= y < H and 0 <= x < W and g[y, x] == 9:
            # click is at the top-left corner of a 6x6 block of 9s
            if y + 6 <= H and x + 6 <= W:
                block = g[y:y + 6, x:x + 6]
                if np.all(block == 9):
                    g[y:y + 6, x:x + 6] = 8
                    _fill_bar(g)
    return g

def is_level_complete(grid):
    g = np.asarray(grid)
    # complete when no 6x6 block of color 9 remains
    H, W = g.shape
    for r in range(H - 5):
        for c in range(W - 5):
            if np.all(g[r:r + 6, c:c + 6] == 9):
                return False
    return True