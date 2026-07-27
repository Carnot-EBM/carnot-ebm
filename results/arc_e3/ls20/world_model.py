import numpy as np

def engine(grid, action, data):
    if action == 3:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        grid = grid.copy()
        grid[py, px] = 5
        return grid
    elif action == 2:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        grid = grid.copy()
        grid[py, px] = 11
        return grid
    return grid

def is_level_complete(grid):
    return False