import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    if action == 4:
        # shift the 0-cup patterns right by 7 (rough guess)
        out = g.copy()
        # find 0 cells in bottom region
        for r in range(H):
            for c in range(W-7):
                if g[r, c] == 0:
                    pass
        # placeholder: do nothing meaningful yet
        return out
    return g

def is_level_complete(grid):
    return False
