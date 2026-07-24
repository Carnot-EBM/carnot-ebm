import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        if grid[py, px] == 14:
            grid[py, px] = 15
        return grid
    return grid

def is_level_complete(grid):
    return np.all(grid == 15)