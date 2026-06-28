import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        if logical_y == 0:
            grid[logical_y, logical_x] = 5
            grid[logical_y + 1, logical_x] = 5
            grid[logical_y + 2, logical_x] = 5
            grid[logical_y + 3, logical_x] = 5
            grid[logical_y + 4, logical_x] = 5
    return grid

def is_level_complete(grid):
    return False