import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        grid = grid.copy()
        grid[py, px] = 5
        return grid
    elif action == 2:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        grid = grid.copy()
        grid[py, px] = 5
        return grid
    elif action == 3:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        grid = grid.copy()
        grid[py, px] = 5
        return grid
    elif action == 5:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        grid = grid.copy()
        grid[py, px] = 5
        return grid
    else:
        return grid

def is_level_complete(grid):
    H, W = grid.shape
    return np.array_equal(grid, np.zeros((H, W), dtype=int))