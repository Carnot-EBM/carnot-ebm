import numpy as np

def engine(grid, action, data):
    if action == 1:
        grid[63, 62] = 11
        grid[63, 60] = 11
        grid[63, 58] = 11
        grid[63, 56] = 11
    elif action == 2:
        grid[63, 54] = 11
        grid[63, 52] = 11
    elif action == 6:
        if data:
            px, py = data['x'], data['y']
            grid[py, px] = 11
    return grid

def is_level_complete(grid):
    return False