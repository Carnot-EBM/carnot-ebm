import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        grid[py, px] = 10
        return grid
    elif action == 4:
        grid[40, 10] = 3
        grid[41, 10] = 2
        return grid
    elif action == 3:
        grid[40, 10] = 2
        grid[41, 10] = 2
        return grid
    else:
        return grid

def is_level_complete(grid):
    return False