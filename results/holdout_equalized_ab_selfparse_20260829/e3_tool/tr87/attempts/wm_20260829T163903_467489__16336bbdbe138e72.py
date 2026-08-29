import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    if action == 4:
        # shift the 0-frame right by 7
        mask = (g == 0)
        if mask.any():
            newg = g.copy()
            # clear old
            newg[mask] = 3
            # place new
            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                nx = x + 7
                if 0 <= nx < g.shape[1]:
                    newg[y, nx] = 0
            g = newg
    return g

def is_level_complete(grid):
    return False
