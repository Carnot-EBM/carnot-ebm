import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        if grid[py, px] == 0:
            grid[py, px] = 5
    return grid

def is_level_complete(grid):
    return False