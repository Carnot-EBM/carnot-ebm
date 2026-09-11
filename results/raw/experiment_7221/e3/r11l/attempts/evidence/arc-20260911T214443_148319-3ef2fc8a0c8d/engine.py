import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    if action == 6 and data is not None:
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        # click at logical (row=y, col=x)
        r, c = y, x
        if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
            # rough hypothesis: clicking a color-6 cell fills the top-left border cell with background
            if g[r, c] == 6:
                g[0, 0] = 5
    return g

def is_level_complete(grid):
    return False
