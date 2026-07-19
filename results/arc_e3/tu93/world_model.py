import numpy as np

def engine(grid, action, data):
    new_grid = grid.copy()
    if action == 4 and data is not None:
        px, py = data['x'], data['y']
        new_grid[py, px] = 0
    return new_grid

def is_level_complete(grid):
    return grid[63, 63] == 6