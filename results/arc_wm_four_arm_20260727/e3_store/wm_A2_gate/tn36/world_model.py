import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if not (0 <= py < grid.shape[0] and 0 <= px < grid.shape[1]):
            return grid
        new_grid = grid.copy()
        new_grid[py, px] = 1
        return new_grid
    return grid

def is_level_complete(grid):
    return False