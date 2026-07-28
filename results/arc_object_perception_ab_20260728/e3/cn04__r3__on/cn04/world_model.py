import numpy as np

def engine(grid, action, data):
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        if logical_y < 0 or logical_y >= 64 or logical_x < 0 or logical_x >= 64:
            return grid
        grid = grid.copy()
        grid[logical_y, logical_x] = 12
        return grid
    return grid

def is_level_complete(grid):
    return np.all(grid == 12)