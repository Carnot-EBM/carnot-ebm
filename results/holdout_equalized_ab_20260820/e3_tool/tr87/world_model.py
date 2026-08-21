import numpy as np

def engine(grid, action, data):
    g = grid.astype(int).copy()
    H, W = g.shape
    if action == 4:
        # shift the 0-pattern right by 7
        mask = (g == 0)
        if mask.any():
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            r0, r1 = int(rows.min()), int(rows.max())
            c0, c1 = int(cols.min()), int(cols.max())
            block = g[r0:r1+1, c0:c1+1].copy()
            # clear old to 3
            g[r0:r1+1, c0:c1+1] = 3
            nc0, nc1 = c0 + 7, c1 + 7
            if nc1 < W:
                g[r0:r1+1, nc0:nc1+1] = block
    elif action in (1, 2):
        # swap 5 and 7 in the lower box region
        # find the box: rows where 7 and 5 coexist in a contiguous band
        # Use the region bounded by the 7-frame (obj48)
        # General: find rows/cols of the 5/7 box
        mask57 = (g == 5) | (g == 7)
        if mask57.any():
            rows = np.where(mask57.any(axis=1))[0]
            cols = np.where(mask57.any(axis=0))[0]
            r0, r1 = int(rows.min()), int(rows.max())
            c0, c1 = int(cols.min()), int(cols.max())
            sub = g[r0:r1+1, c0:c1+1].copy()
            tmp = sub.copy()
            tmp[sub == 5] = 7
            tmp[sub == 7] = 5
            g[r0:r1+1, c0:c1+1] = tmp
    return g

def is_level_complete(grid):
    return False
