import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        # Apply toggle at (py, px)
        grid_copy = grid.copy()
        grid_copy[py, px] = 13
        return grid_copy
    return grid

def is_level_complete(grid):
    return False