import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    if action == 4:
        # shift the 15-liquid region right by 1 (rough hypothesis)
        mask = (g == 15)
        g2 = g.copy()
        # move 15s right where target is 5
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if mask[r, c]:
                    if c + 1 < W and g2[r, c + 1] == 5:
                        g2[r, c + 1] = 15
                        g2[r, c] = 5
    return g

def is_level_complete(grid):
    return False
