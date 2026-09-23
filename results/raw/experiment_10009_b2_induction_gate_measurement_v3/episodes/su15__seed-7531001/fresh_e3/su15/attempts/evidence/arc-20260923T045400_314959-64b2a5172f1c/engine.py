import numpy as np


def engine(grid, action, data):
    grid = grid.copy()
    if action == 6 and data is not None:
        px, py = int(data['x']), int(data['y'])
        # Determine which "slot" was clicked (0-7) based on x coordinate
        slot = min(7, max(0, px // 8))
        # Row 63 starts all zeros; fill 2 cells per click starting from right
        base_col = 62 - slot * 2
        if base_col >= 0:
            grid[63, base_col] = 5
            grid[63, base_col + 1] = 5
    return grid


def is_level_complete(grid):
    return bool(np.all(grid[63, :] == 5))