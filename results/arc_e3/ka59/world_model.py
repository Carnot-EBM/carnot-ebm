import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if grid[py, px] == 0:
            grid[py, px] = 1
    elif action == 4:
        if grid[30, 18] == 0:
            grid[30, 18] = 15
            grid[31, 18] = 10
            grid[32, 18] = 15
    elif action == 1:
        if grid[27, 21] == 0:
            grid[27, 21] = 10
            grid[28, 21] = 10
            grid[29, 21] = 10
            grid[30, 21] = 15
            grid[31, 21] = 15
            grid[32, 21] = 15
    return grid

def is_level_complete(grid):
    return False