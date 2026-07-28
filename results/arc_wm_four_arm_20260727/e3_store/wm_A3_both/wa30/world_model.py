import numpy as np

def engine(grid, action, data):
    if action == 3:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if grid[py, px] == 0:
            grid[py, px] = 5
            if py == 63:
                grid[py, px] = 0
    elif action == 2:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if grid[py, px] == 0:
            grid[py, px] = 5
            if py == 63:
                grid[py, px] = 0
    return grid

def is_level_complete(grid):
    return False