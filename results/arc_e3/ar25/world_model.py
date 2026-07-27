import numpy as np

def engine(grid, action, data):
    if action == 7:
        return grid.copy()
    if action == 6:
        return grid.copy()
    if action == 2:
        return grid.copy()
    if action == 3:
        return grid.copy()
    if action == 4:
        return grid.copy()
    return grid.copy()

def is_level_complete(grid):
    return False