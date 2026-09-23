import numpy as np


def engine(grid, action, data):
    grid = grid.copy()
    if action == 6 and data is not None:
        px, py = int(data['x']), int(data['y'])
        # Determine which "slot" was clicked (left or right)
        slot = 0 if px < 32 else 1
        base_col = 52 + slot * 4
        for i in range(2):
            col = base_col + i
            if grid[63, col] != 5:
                grid[63, col] = 5
                break
    return grid


def is_level_complete(grid):
    # Win when all 8 slots at row 63 are filled with color 5
    return bool(np.all(grid[63, :] == 5))