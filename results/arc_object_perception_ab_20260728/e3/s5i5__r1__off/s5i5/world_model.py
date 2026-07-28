import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        logical_x, logical_y = px, py
        if logical_y < 64:
            grid[logical_y, logical_x] = 14
            grid[logical_y, logical_x + 1] = 14
            grid[logical_y, logical_x + 2] = 14
            grid[logical_y, logical_x + 3] = 14
            grid[63, logical_x - 1] = 4
            grid[63, logical_x - 2] = 4
            grid[63, logical_x - 3] = 4
            grid[63, logical_x - 4] = 4
    return grid

def is_level_complete(grid):
    return np.all(grid == 5)