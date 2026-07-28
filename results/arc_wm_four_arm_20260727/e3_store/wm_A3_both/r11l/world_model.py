import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        if py < 0 or py >= h or px < 0 or px >= w:
            return new_grid
        if new_grid[py, px] == 0:
            new_grid[py, px] = 3
        else:
            new_grid[py, px] = 0
        return new_grid
    return grid

def is_level_complete(grid):
    return False