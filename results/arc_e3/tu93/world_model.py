import numpy as np

def engine(grid, action, data):
    grid = grid.copy()
    if action == 1:
        if data is not None:
            px, py = data['x'], data['y']
            grid[py, px] = 0
        else:
            grid[63, 63] = 0
    return grid

def is_level_complete(grid):
    return False