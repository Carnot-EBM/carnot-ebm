import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        if 0 <= px < W and 0 <= py < H:
            # Toggle cell at (py, px)
            if grid[py, px] == 5:
                grid[py, px] = 13
            else:
                grid[py, px] = 5
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if the grid matches the win state pattern
    # Rows 0-8: all 5s
    for i in range(9):
        if not np.all(grid[i] == 5):
            return False
    # Rows 9-26: specific pattern
    for i in range(9, 27):
        if not np.all(grid[i] == 5):
            return False
    # Rows 27-35: specific pattern
    for i in range(27, 36):
        if not np.all(grid[i] == 5):
            return False
    # Rows 36-44: specific pattern
    for i in range(36, 45):
        if not np.all(grid[i] == 5):
            return False
    # Rows 45-53: all 5s
    for i in range(45, 54):
        if not np.all(grid[i] == 5):
            return False
    # Rows 54-63: specific pattern
    for i in range(54, 64):
        if not np.all(grid[i] == 5):
            return False
    return True