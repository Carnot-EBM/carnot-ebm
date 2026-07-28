import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        return grid
    return grid

def is_level_complete(grid):
    return True