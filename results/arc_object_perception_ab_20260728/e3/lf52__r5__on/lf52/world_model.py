import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        grid = grid.copy()
        grid[py, px] = 5
        return grid
    return grid

def is_level_complete(grid):
    return np.all(grid[7:, :] == 5)