import numpy as np

def engine(grid, action, data):
    if action == 1:
        # Action 1: Shift the inner pattern down by 1 row
        h, w = grid.shape
        new_grid = grid.copy()
        # Shift inner region down
        for r in range(h - 1):
            for c in range(w):
                if grid[r, c] != 5:
                    new_grid[r + 1, c] = grid[r, c]
        # Clear the top row of the inner region
        for c in range(11, w - 1):
            new_grid[11, c] = 5
        return new_grid
    elif action == 2:
        # Action 2: Shift the inner pattern up by 1 row
        h, w = grid.shape
        new_grid = grid.copy()
        # Shift inner region up
        for r in range(1, h):
            for c in range(w):
                if grid[r, c] != 5:
                    new_grid[r - 1, c] = grid[r, c]
        # Clear the bottom row of the inner region
        for c in range(11, w - 1):
            new_grid[h - 1, c] = 5
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    h, w = grid.shape
    # Check for the specific win state pattern
    # The win state has a specific pattern in the inner region
    # Based on the observed transitions, the win state is when the inner region is fully filled
    # with a specific pattern
    # Check if the grid matches the win state
    for r in range(11, h - 1):
        for c in range(11, w - 1):
            if grid[r, c] == 5:
                return False
    return True