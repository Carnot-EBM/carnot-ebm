import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        if logical_y < 0 or logical_y >= grid.shape[0] or logical_x < 0 or logical_x >= grid.shape[1]:
            return grid
        grid_copy = grid.copy()
        grid_copy[logical_y, logical_x] = 0
        return grid_copy
    return grid

def is_level_complete(grid):
    if grid is None:
        return False
    return np.all(grid == 4)