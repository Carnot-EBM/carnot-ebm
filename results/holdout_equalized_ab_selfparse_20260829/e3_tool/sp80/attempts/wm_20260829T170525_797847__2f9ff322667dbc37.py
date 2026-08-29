import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    if action == 4:
        # find color-9 object
        mask = (g == 9)
        if mask.any():
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            r0, r1 = rows.min(), rows.max()
            c0, c1 = cols.min(), cols.max()
            block = g[r0:r1+1, c0:c1+1].copy()
            # move right by 4
            shift = 4
            nc0 = c0 + shift
            nc1 = c1 + shift
            if nc1 < W:
                # clear old
                g[r0:r1+1, c0:c1+1] = 12
                # place new
                g[r0:r1+1, nc0:nc1+1] = block
        # shrink top color-14 bar by 2 from the right
        top = (g[0] == 14)
        if top.any():
            idx = np.where(top)[0]
            right = idx.max()
            # remove 2 cells from the right end of the contiguous bar
            # find contiguous run ending at right
            run_end = right
            run_start = right
            while run_start - 1 >= 0 and g[0, run_start-1] == 14:
                run_start -= 1
            # remove up to 2 cells from the right
            for k in range(2):
                pos = run_end - k
                if pos >= run_start:
                    g[0, pos] = 0
    return g

def is_level_complete(grid):
    return False
