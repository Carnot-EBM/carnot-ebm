import numpy as np

def engine(grid, action, data):
    if action == 6 and data:
        px, py = data['x'], data['y']
        grid[py, px] = 8
        return grid

    return grid

def is_level_complete(grid):
    return np.all(grid == 4)