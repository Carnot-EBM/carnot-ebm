import numpy as np

def engine(grid, action, data):
    if action == 1:
        grid[63, 63] = 12
    elif action == 2:
        grid[63, 63] = 12
    elif action == 3:
        grid[63, 63] = 12
    elif action == 4:
        grid[63, 63] = 12
    elif action == 5:
        grid[63, 63] = 12
    elif action == 6:
        grid[63, 63] = 12
    elif action == 7:
        grid[63, 63] = 12
    return grid

def is_level_complete(grid):
    return grid[63, 63] == 12