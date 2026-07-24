import numpy as np

def engine(grid, action, data):
    if action == 3:
        grid[0, 0] = 1
        return grid
    if action == 6 and data is not None:
        px, py = data['x'], data['y']
        grid[py, px] = 1
        return grid
    return grid

def is_level_complete(grid):
    return False