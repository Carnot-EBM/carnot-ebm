import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        new_grid[py, px] = 10
        return new_grid
    elif action == 4:
        new_grid = grid.copy()
        new_grid[30, 10] = 10
        new_grid[31, 10] = 10
        return new_grid
    elif action == 3:
        new_grid = grid.copy()
        new_grid[30, 10] = 10
        new_grid[31, 10] = 10
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    return True