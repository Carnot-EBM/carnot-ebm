import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    H, W = g.shape
    if action == 4:
        # move the 15 blob down-right by 1, leaving a 2 trail behind it
        mask = (g == 15)
        if mask.any():
            new = np.zeros_like(g)
            new[1:, 1:] = mask[:-1, :-1]
            # cells vacated by the blob become 2 (trail)
            vac = mask & ~np.roll(np.roll(new, 1, axis=0), 1, axis=1)
            g[vac] = 2
            g[new] = 15
    elif action == 2:
        # move the 15 blob down-left by 1, leaving a 2 trail behind it
        mask = (g == 15)
        if mask.any():
            new = np.zeros_like(g)
            new[1:, :-1] = mask[:-1, 1:]
            vac = mask & ~np.roll(np.roll(new, 1, axis=0), -1, axis=1)
            g[vac] = 2
            g[new] = 15
    return g

def is_level_complete(grid):
    return False