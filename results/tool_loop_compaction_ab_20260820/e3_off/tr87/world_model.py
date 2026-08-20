import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    H, W = g.shape
    if action == 4:
        # Find all 0-cells and shift them right by 7
        zeros = np.argwhere(g == 0)
        if len(zeros):
            changes = {}
            for (r, c) in zeros:
                changes[(r, c)] = 3
                if c + 7 < W:
                    changes[(r, c + 7)] = 0
            for (r, c), v in changes.items():
                if 0 <= r < H and 0 <= c < W:
                    g[r, c] = v
    elif action in (1, 2):
        # Find the 7-bordered box in the lower region
        # 7-cells in lower half
        lower = g[H//2:]
        sevens = np.argwhere(lower == 7)
        if len(sevens):
            r0 = sevens[:, 0].min() + H//2
            r1 = sevens[:, 0].max() + H//2
            c0 = sevens[:, 1].min()
            c1 = sevens[:, 1].max()
            # Swap 5<->7 in the interior (exclude border)
            interior = g[r0+1:r1, c0+1:c1]
            interior = np.where(interior == 5, 7, np.where(interior == 7, 5, interior))
            g[r0+1:r1, c0+1:c1] = interior
        # Move the 4 in the last row
        last_row = H - 1
        idx = np.where(g[last_row] == 4)[0]
        if len(idx):
            c = idx[0]
            g[last_row, c] = 1
            if action == 2:
                g[last_row, (c - 1) % W] = 4
            else:
                g[last_row, (c + 1) % W] = 4
    return g

def is_level_complete(grid):
    return False
