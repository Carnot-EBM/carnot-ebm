import numpy as np

def engine(grid, action, data):
    if action == 3:
        return grid
    elif action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if grid[py, px] == 10:
            return grid
        new_grid = grid.copy()
        new_grid[py, px] = 0
        return new_grid
    return grid

def is_level_complete(grid):
    return False