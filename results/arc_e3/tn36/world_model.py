import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        if 0 <= py < grid.shape[0] and 0 <= px < grid.shape[1]:
            new_grid[py, px] = 3
        return new_grid
    return grid

def is_level_complete(grid):
    return False