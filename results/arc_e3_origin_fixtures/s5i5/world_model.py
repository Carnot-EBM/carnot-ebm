import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        grid = grid.copy()
        grid[py, px] = 11
        grid[py, px + 1] = 11
        grid[py, px + 2] = 11
        grid[py, px + 3] = 11
        grid[63, 63] = 11
        return grid
    return grid

def is_level_complete(grid):
    return False