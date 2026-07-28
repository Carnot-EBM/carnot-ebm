import numpy as np

def engine(grid, action, data):
    if action == 6:
        return grid
    if action == 3:
        return grid
    if action == 4:
        return grid
    if action == 7:
        return grid
    if action == 1:
        return grid
    return grid

def is_level_complete(grid):
    return False