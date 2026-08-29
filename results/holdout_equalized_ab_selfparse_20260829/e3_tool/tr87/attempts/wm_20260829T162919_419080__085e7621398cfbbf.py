import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    H, W = g.shape
    if action == 4:
        # move the 0-hole frame RIGHT by 7 columns
        for r in range(H):
            cols = np.where(g[r] == 0)[0]
            if len(cols) > 0:
                g[r, cols] = 3
                g[r, cols + 7] = 0
    elif action == 1:
        # move 5s right by 1
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if g[r, c] == 5:
                    g[r, c + 1] = 5
                    g[r, c] = 7
    elif action == 2:
        # move 5s left by 1
        for r in range(H):
            for c in range(W):
                if g[r, c] == 5:
                    g[r, c - 1] = 5
                    g[r, c] = 7
    return g

def is_level_complete(grid):
    return False
