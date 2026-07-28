import numpy as np

def engine(grid, action, data):
    if action == 1:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        new_grid[py, px] = 5
        return new_grid
    elif action == 3:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        new_grid[py, px] = 10
        return new_grid
    elif action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        new_grid[py, px] = 12
        return new_grid
    return grid

def is_level_complete(grid):
    if grid is None:
        return False
    if grid.shape != (64, 64):
        return False
    return np.all(grid == 5)