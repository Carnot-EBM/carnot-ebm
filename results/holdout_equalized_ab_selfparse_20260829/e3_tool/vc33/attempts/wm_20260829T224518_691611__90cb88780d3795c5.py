import numpy as np

def engine(grid, action, data):
    g = grid.astype(int).copy()
    H, W = g.shape
    if action == 6 and data is not None:
        px, py = int(data.get('x', 0)), int(data.get('y', 0))
        # Hypothesis: clicking moves the 4/11 diamond left by 4,
        # shifts a 4-wide column of 3, and updates top row.
        # Move the 4/11 diamond (colors 4 and 11 in the lower area) left by 4.
        # Find the diamond: connected region of 4s and 11s in rows 40+.
        # Simple approach: shift all 4 and 11 cells in rows 40..H-1 left by 4.
        for r in range(40, H):
            for c in range(W - 1, 3, -1):
                if g[r, c] in (4, 11):
                    g[r, c - 4] = g[r, c]
                    g[r, c] = 0
    return g

def is_level_complete(grid):
    return False
