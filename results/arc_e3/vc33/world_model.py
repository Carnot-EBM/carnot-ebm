import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        new_grid[py, px] = 15
        return new_grid
    return grid

def is_level_complete(grid):
    return True