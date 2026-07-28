import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if grid[py, px] == 0:
            grid[py, px] = 15
            return grid
        return grid
    return grid

def is_level_complete(grid):
    return grid[0, :] == 15 and grid[63, :] == 0