import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        grid[py, px] = 10
        return grid
    elif action == 1:
        grid[0, 1] = 10
        return grid
    else:
        return grid

def is_level_complete(grid):
    return False