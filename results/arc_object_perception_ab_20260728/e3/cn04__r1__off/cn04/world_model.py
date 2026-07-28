import numpy as np

def engine(grid, action, data):
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        grid = grid.copy()
        grid[py, px] = 12
        return grid
    return grid

def is_level_complete(grid):
    return np.all(grid == 12)