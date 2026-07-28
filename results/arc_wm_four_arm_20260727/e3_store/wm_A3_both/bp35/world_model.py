import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        if 0 <= px < h and 0 <= py < w:
            grid_copy = grid.copy()
            grid_copy[px, py] = 1
            return grid_copy
        return grid
    return grid

def is_level_complete(grid):
    return False