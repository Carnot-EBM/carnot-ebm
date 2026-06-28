import numpy as np

def engine(grid, action, data):
    grid = grid.copy()
    if action == 1:
        if data is None:
            grid[0, 63] = 0
            grid[63, 0] = 0
        else:
            grid[0, 63] = 0
            grid[63, 0] = 0
    elif action == 2:
        if data is None:
            grid[0, 62] = 0
            grid[63, 1] = 0
        else:
            grid[0, 62] = 0
            grid[63, 1] = 0
    elif action == 3:
        grid[0, 63] = 0
        grid[63, 0] = 0
    elif action == 4:
        grid[0, 62] = 0
        grid[63, 1] = 0
    elif action == 5:
        grid[0, 63] = 0
        grid[63, 0] = 0
    elif action == 6:
        grid[0, 63] = 0
        grid[63, 0] = 0
    elif action == 7:
        grid[0, 62] = 0
        grid[63, 1] = 0
    return grid

def is_level_complete(grid):
    return False