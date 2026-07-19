import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if grid[py, px] == 3:
            grid[py, px] = 4
            grid[py, px + 1] = 4
            grid[py, px + 2] = 4
            grid[py, px + 3] = 4
    return grid

def is_level_complete(grid):
    return grid[63, 63] == 4