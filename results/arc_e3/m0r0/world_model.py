import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        if 0 <= logical_y < grid.shape[0] and 0 <= logical_x < grid.shape[1]:
            if grid[logical_y, logical_x] == 5:
                grid_copy = grid.copy()
                grid_copy[logical_y, logical_x] = 0
                grid_copy[0, grid.shape[1] - 1] = 0
                grid_copy[grid.shape[0] - 1, 0] = 0
                return grid_copy
    return grid

def is_level_complete(grid):
    return False