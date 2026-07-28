import numpy as np

def engine(grid, action, data):
    if action == 1:
        # Action 1: Toggle a specific region (rows 18-29, col 11)
        rows = np.arange(18, 30)
        cols = np.arange(11, 17)
        grid[rows, cols] = 5 - grid[rows, cols]
        return grid
    elif action == 3:
        # Action 3: Toggle a specific region (rows 19-22, cols 36-42)
        rows = np.arange(19, 23)
        cols = np.arange(36, 43)
        grid[rows, cols] = 5 - grid[rows, cols]
        return grid
    elif action == 4:
        # Action 4: Toggle a specific region (rows 20-21, cols 23-46)
        rows = np.arange(20, 22)
        cols = np.arange(23, 47)
        grid[rows, cols] = 5 - grid[rows, cols]
        return grid
    else:
        return grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # Win state has specific structure in rows 6-47
    # Check if rows 6-47 have the pattern: 5x11, 4x42, 5x11
    for i in range(6, 48):
        row = grid[i]
        # Check if row matches the pattern
        if not np.array_equal(row, np.concatenate([np.full(11, 5), np.full(42, 4), np.full(11, 5)])):
            return False
    return True