import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if grid[py, px] == 0:
            grid[py, px] = 3
        return grid
    elif action == 1:
        grid = grid.copy()
        grid[38, 10] = 14
        grid[39, 10] = 14
        grid[40, 10] = 2
        grid[41, 10] = 2
        return grid
    elif action == 2:
        grid = grid.copy()
        grid[38, 10] = 2
        grid[39, 10] = 2
        grid[40, 10] = 14
        grid[41, 10] = 14
        return grid
    else:
        return grid

def is_level_complete(grid):
    return False