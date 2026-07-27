import numpy as np

def engine(grid, action, data):
    grid = grid.copy()
    if action == 6:
        px, py = data['x'], data['y']
        grid[py, px] = 7
    elif action == 3:
        if data is None:
            return grid
        grid[py, px] = 5
    elif action == 2:
        if data is None:
            return grid
        grid[py, px] = 7
    return grid

def is_level_complete(grid):
    return False