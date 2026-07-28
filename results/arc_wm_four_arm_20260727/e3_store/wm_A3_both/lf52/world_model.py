import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        if grid[py, px] == 0:
            grid[py, px] = 10
        else:
            grid[py, px] = 0
        return grid
    elif action == 1:
        if grid[0, 1] == 0:
            grid[0, 1] = 10
        else:
            grid[0, 1] = 0
        return grid
    else:
        return grid

def is_level_complete(grid):
    return False