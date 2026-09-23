import numpy as np


def engine(grid, action, data):
    g = grid.copy()
    if action == 6 and data is not None:
        px, py = int(data['x']), int(data['y'])
        x, y = px // 1, py // 1
        if 0 <= y < g.shape[0] and 0 <= x < g.shape[1]:
            g[y, x] = 4
    return g


def is_level_complete(grid):
    return False