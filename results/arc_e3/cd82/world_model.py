import numpy as np

def engine(grid, action, data):
    if action == 1:
        return grid
    if action == 3:
        return grid
    if action == 5:
        return grid
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        grid = grid.copy()
        grid[py, px] = 5
        return grid
    return grid

def is_level_complete(grid):
    return False