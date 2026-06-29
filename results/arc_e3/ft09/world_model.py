import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        if 0 <= logical_y < grid.shape[0] and 0 <= logical_x < grid.shape[1]:
            grid = grid.copy()
            grid[logical_y, logical_x] = 0
            return grid
    return grid.copy()

def is_level_complete(grid):
    return False