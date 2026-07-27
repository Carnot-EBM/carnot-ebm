import numpy as np

def engine(grid, action, data):
    if action == 6:
        px = data['x']
        py = data['y']
        new_grid = grid.copy()
        new_grid[py, px] = 11
        new_grid[py - 1, px] = 11
        new_grid[py - 2, px] = 11
        new_grid[py - 3, px] = 11
        new_grid[py - 4, px] = 11
        new_grid[py - 5, px] = 11
        new_grid[py - 6, px] = 11
        new_grid[py - 7, px] = 11
        new_grid[py - 8, px] = 11
        new_grid[py - 9, px] = 11
        new_grid[py - 10, px] = 11
        new_grid[py - 11, px] = 11
        new_grid[py - 12, px] = 11
        new_grid[py - 13, px] = 14
        return new_grid
    return grid

def is_level_complete(grid):
    return False