import numpy as np

def engine(grid, action, data):
    if action == 3:
        if grid[38, 17] == 4 and grid[39, 17] == 4:
            grid[38, 17] = 6
            grid[39, 17] = 6
    return grid

def is_level_complete(grid):
    return False