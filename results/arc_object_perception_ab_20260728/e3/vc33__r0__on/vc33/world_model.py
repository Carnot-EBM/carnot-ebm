import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        logical_x, logical_y = px, py
        if logical_x < 0 or logical_x >= 64 or logical_y < 0 or logical_y >= 64:
            return grid
        target_color = 7
        grid[logical_y, logical_x] = target_color
        return grid
    return grid

def is_level_complete(grid):
    if grid[0, 0] != 7:
        return False
    for y in range(64):
        for x in range(64):
            if grid[y, x] != 0 and grid[y, x] != 7:
                return False
    return True