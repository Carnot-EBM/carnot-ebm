import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        grid[py, px] = 15
        return grid
    return grid

def is_level_complete(grid):
    return np.all(grid == 0)