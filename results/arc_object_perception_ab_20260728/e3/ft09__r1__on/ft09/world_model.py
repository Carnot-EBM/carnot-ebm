import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        new_grid[py, px] = 8
        return new_grid
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    if h != 64 or w != 64:
        return False
    for r in range(64):
        if grid[r, 0] != 4:
            return False
    for r in range(64):
        if grid[r, 0:60].min() != 4 or grid[r, 60:64].min() != 9:
            return False
    return True