import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        if logical_y < 64:
            for i in range(64):
                grid[logical_y, i] = 5
    return grid

def is_level_complete(grid):
    return False