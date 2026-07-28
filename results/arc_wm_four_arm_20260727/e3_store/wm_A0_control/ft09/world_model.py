import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 0 or py >= grid.shape[0] or px < 0 or px >= grid.shape[1]:
            return grid
        if grid[py, px] == 1:
            return grid
        new_grid = grid.copy()
        new_grid[py, px] = 15
        return new_grid
    return grid

def is_level_complete(grid):
    return False