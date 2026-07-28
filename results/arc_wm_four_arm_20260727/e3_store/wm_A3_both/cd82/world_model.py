import numpy as np

def engine(grid, action, data):
    grid = grid.copy()
    if action == 1:
        grid[63, 63] = 5
    elif action == 3:
        grid[63, 62] = 5
    elif action == 5:
        grid[63, 61] = 5
    elif action == 6:
        if data:
            grid[data['y'], data['x']] = 5
    return grid

def is_level_complete(grid):
    return False