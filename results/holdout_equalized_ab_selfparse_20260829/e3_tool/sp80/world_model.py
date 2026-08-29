import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    H, W = g.shape
    if action == 4:
        # find the color-9 block (the active object)
        mask = (g == 9)
        if mask.any():
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            r0, r1 = int(rows.min()), int(rows.max())
            c0, c1 = int(cols.min()), int(cols.max())
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            step = 4
            new_c0 = c0 + step
            new_c1 = c1 + step
            if new_c1 < W:
                # clear old, set new
                g[r0:r1+1, c0:c1+1] = 12
                g[r0:r1+1, new_c0:new_c1+1] = 9
        # deplete 2 cells from the right of the color-14 top bar
        top = g[0]
        # find rightmost run of 14
        idx = np.where(top == 14)[0]
        if idx.size:
            right = int(idx.max())
            # consume up to 2 cells at the right end of the contiguous 14 run
            cnt = 0
            c = right
            while c >= 0 and top[c] == 14 and cnt < 2:
                top[c] = 0
                c -= 1
                cnt += 1
            g[0] = top
    return g

def is_level_complete(grid):
    g = np.array(grid)
    # placeholder: no 9 block remaining? (unknown)
    return False
