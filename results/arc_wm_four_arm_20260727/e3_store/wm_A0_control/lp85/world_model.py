import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        logical_x, logical_y = px, py
        if logical_y < 64 and logical_x < 64:
            grid[logical_y, logical_x] = 14
            grid[logical_y, logical_x + 1] = 14
            grid[logical_y, logical_x + 2] = 14
            grid[logical_y, logical_x + 3] = 14
    return grid

def is_level_complete(grid):
    return False