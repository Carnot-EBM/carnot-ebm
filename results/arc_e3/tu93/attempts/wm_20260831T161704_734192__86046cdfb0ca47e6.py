import numpy as np

def engine(grid, action, data):
    g = np.array(grid, copy=True)
    H, W = g.shape
    # The color-6 bottom row is the player / progress bar.
    # Directional actions shift it; the trailing edge reverts to background (0).
    # ACTION4 = right: the rightmost 6 cell becomes 0.
    if action == 4:
        # find rightmost 6 in the bottom row
        row = g[H-1]
        cols = np.where(row == 6)[0]
        if len(cols) > 0:
            g[H-1, cols[-1]] = 0
    return g

def is_level_complete(grid):
    return False
