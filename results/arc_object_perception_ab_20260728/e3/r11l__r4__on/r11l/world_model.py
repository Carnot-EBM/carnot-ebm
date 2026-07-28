import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        new_grid[py, px] = 5
        return new_grid
    return grid

def is_level_complete(grid):
    return False