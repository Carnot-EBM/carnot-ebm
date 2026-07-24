import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        h, w = grid.shape
        if 0 <= px < w and 0 <= py < h:
            grid = grid.copy()
            grid[py, px] = 14
            return grid
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    return np.all(grid == 14)