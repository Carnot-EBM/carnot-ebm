import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        grid = grid.copy()
        grid[py, px] = 1
        return grid
    return grid

def is_level_complete(grid):
    return np.array_equal(grid, np.zeros((64, 64), dtype=int))