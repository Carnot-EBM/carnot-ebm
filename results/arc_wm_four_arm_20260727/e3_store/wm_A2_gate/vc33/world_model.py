import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        if 0 <= py < 64 and 0 <= px < 64:
            grid[py, px] = 3
    return grid

def is_level_complete(grid):
    return np.array_equal(grid, np.zeros((64, 64), dtype=int))