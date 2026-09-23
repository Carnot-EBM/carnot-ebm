import numpy as np


def engine(grid, action, data):
    grid = grid.copy()
    if action == 6 and data is not None:
        px, py = int(data['x']), int(data['y'])
        x, y = px // 1, py // 1
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            # Toggle color at clicked cell between 3 and 4 (or similar simple toggle)
            if grid[y, x] == 3:
                grid[y, x] = 4
            elif grid[y, x] == 4:
                grid[y, x] = 3
    return grid


def is_level_complete(grid):
    return False