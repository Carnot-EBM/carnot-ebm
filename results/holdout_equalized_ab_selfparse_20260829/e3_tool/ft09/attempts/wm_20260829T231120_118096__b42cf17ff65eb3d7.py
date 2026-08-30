import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    H, W = g.shape
    if action == 6 and data is not None:
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        # click at top-left corner of a 6x6 block of color 9 -> convert to 8
        r0, c0 = y, x
        if 0 <= r0 and 0 <= c0 and r0 + 6 <= H and c0 + 6 <= W:
            block = g[r0:r0+6, c0:c0+6]
            if np.all(block == 9):
                g[r0:r0+6, c0:c0+6] = 8
                # counter: fill 2 more cells of color 11 in bottom row, right-to-left
                n11 = int(np.sum(g[-1] == 11))
                c = W - 1 - n11
                if c >= 1:
                    g[-1, c-1:c+1] = 11
    return g

def is_level_complete(grid):
    g = np.array(grid)
    # level complete when the bottom counter row is fully filled with 11
    return bool(np.all(g[-1] == 11))
