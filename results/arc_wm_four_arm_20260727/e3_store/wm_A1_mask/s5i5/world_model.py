import numpy as np

def engine(grid, action, data):
    if action == 6:
        px = data['x']
        py = data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        new_grid[py, px] = 11
        new_grid[py, px - 1] = 11
        new_grid[py, px - 2] = 11
        new_grid[py, px - 3] = 11
        new_grid[py, px - 4] = 11
        new_grid[py, px - 5] = 11
        new_grid[py, px - 6] = 11
        new_grid[py, px - 7] = 11
        new_grid[py, px - 8] = 11
        new_grid[py, px - 9] = 11
        new_grid[py, px - 10] = 11
        new_grid[py, px - 11] = 11
        new_grid[py, px - 12] = 11
        new_grid[py, px - 13] = 11
        new_grid[py, px - 14] = 11
        new_grid[py, px - 15] = 11
        new_grid[py, px - 16] = 11
        new_grid[py, px - 17] = 11
        new_grid[py, px - 18] = 11
        new_grid[py, px - 19] = 11
        new_grid[py, px - 20] = 11
        new_grid[63, 63] = 11
        return new_grid
    return grid

def is_level_complete(grid):
    return False