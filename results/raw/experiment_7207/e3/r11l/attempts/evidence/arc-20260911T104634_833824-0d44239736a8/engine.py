import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    H, W = g.shape
    if action == 6 and data is not None:
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        r, c = y, x  # pixel==logical
        if 0 <= r < H and 0 <= c < W:
            # click on the player (color 6): paint the top-left corner to background
            if g[r, c] == 6:
                g[0, 0] = 5
    return g

def is_level_complete(grid):
    return False
