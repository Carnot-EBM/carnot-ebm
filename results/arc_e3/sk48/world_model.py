import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        return grid
    if action in [1, 3, 5, 7]:
        return grid
    if action == 2:
        return grid
    if action == 4:
        return grid
    if action == 6:
        return grid
    return grid

def is_level_complete(grid):
    return False