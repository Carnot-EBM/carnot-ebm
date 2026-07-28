import numpy as np

def engine(grid, action, data):
    if action == 1:
        return grid
    if action == 3:
        return grid
    if action == 5:
        return grid
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        if 0 <= py < h and 0 <= px < w:
            grid = grid.copy()
            grid[py, px] = 5
            return grid
        return grid
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    if h != 64 or w != 64:
        return False
    for i in range(h):
        if np.all(grid[i] == 5):
            continue
        return False
    return True