import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    if action == 6 and data is not None:
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        H, W = g.shape
        # Check if the 6x6 block at (y, x) is all 9s
        if y + 6 <= H and x + 6 <= W:
            block = g[y:y+6, x:x+6]
            if np.all(block == 9):
                g[y:y+6, x:x+6] = 8
                # Update progress bar in last row: add 2 cells of 11, filling from the right
                last_row = H - 1
                n = int(np.sum(g[last_row] == 11))
                n_new = min(n + 2, W)
                # Clear existing 11s and rewrite
                g[last_row, :] = 12
                if n_new > 0:
                    g[last_row, W - n_new:] = 11
    return g

def is_level_complete(grid):
    # Level complete when progress bar is full (all 11s in last row)
    H, W = grid.shape
    return bool(np.all(grid[H-1] == 11))
