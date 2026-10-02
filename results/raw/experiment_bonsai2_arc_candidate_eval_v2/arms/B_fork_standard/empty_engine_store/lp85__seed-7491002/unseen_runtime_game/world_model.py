import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    if action == 6 and data is not None:
        px, py = data['x'], data['y']
        r, c = py // 1, px // 1
        if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
            g[r, c] = 5
    return g

def is_level_complete(grid):
    return False