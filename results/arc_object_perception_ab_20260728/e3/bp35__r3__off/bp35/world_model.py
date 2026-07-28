import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        if px < 0 or px >= w or py < 0 or py >= h:
            return grid
        new_grid = grid.copy()
        new_grid[py, px] = 15
        return new_grid
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    if h != 64 or w != 64:
        return False
    for r in range(h):
        row = grid[r, :]
        if np.any(row == 0):
            return False
        if np.any(row == 15):
            return False
    return True