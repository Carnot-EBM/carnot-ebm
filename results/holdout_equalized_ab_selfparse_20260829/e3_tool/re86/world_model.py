import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    H, W = g.shape
    if action == 4:
        # move the 0 (hole) right by 3, dragging the 9 band with it
        for r in range(H):
            row = g[r]
            z = np.where(row == 0)[0]
            if z.size == 0:
                continue
            c = int(z[0])
            c2 = c + 3
            if c2 >= W:
                continue
            g[r, c] = row[c2]
            g[r, c2] = 0
    elif action == 5:
        # move the 0 (hole) left by 3
        for r in range(H):
            row = g[r]
            z = np.where(row == 0)[0]
            if z.size == 0:
                continue
            c = int(z[0])
            c2 = c - 3
            if c2 < 0:
                continue
            g[r, c] = row[c2]
            g[r, c2] = 0
    elif action == 1:
        # move the 0 (hole) up by 3, dragging the 11 band with it
        for c in range(W):
            col = g[:, c]
            z = np.where(col == 0)[0]
            if z.size == 0:
                continue
            r = int(z[0])
            r2 = r - 3
            if r2 < 0:
                continue
            g[r, c] = col[r2]
            g[r2, c] = 0
    elif action == 2:
        # move the 0 (hole) down by 3
        for c in range(W):
            col = g[:, c]
            z = np.where(col == 0)[0]
            if z.size == 0:
                continue
            r = int(z[0])
            r2 = r + 3
            if r2 >= H:
                continue
            g[r, c] = col[r2]
            g[r2, c] = 0
    return g

def is_level_complete(grid):
    return False
