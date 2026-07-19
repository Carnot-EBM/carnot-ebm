import numpy as np

def engine(grid, action, data):
    if action == 3:
        return grid.copy()
    if action == 6:
        if data is None:
            return grid.copy()
        px, py = data['x'], data['y']
        logical_x, logical_y = px, py
        if logical_x >= grid.shape[1] or logical_y >= grid.shape[0]:
            return grid.copy()
        if grid[logical_y, logical_x] == 0:
            grid[logical_y, logical_x] = 5
            return grid.copy()
        return grid.copy()
    return grid.copy()

def is_level_complete(grid):
    return False