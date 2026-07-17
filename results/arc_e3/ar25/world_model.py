import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] == 5:
                    new_grid[r, c] = grid[r - 1, c]
                    new_grid[r - 1, c] = 5
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H - 1):
                if grid[r, c] == 5:
                    new_grid[r, c] = grid[r + 1, c]
                    new_grid[r + 1, c] = 5
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if grid[r, c] == 5:
                    new_grid[r, c] = grid[r, c - 1]
                    new_grid[r, c - 1] = 5
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    new_grid[r, c] = grid[r, c + 1]
                    new_grid[r, c + 1] = 5
    elif action == 5:
        # Toggle 5 <-> 11
        new_grid = grid.copy()
        new_grid[grid == 5] = 11
        new_grid[grid == 11] = 5
    elif action == 6:
        # Click at data['x'], data['y']
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        if grid[py, px] == 5:
            new_grid[py, px] = 11
        elif grid[py, px] == 11:
            new_grid[py, px] = 5
    elif action == 7:
        # Toggle 5 <-> 9
        new_grid = grid.copy()
        new_grid[grid == 5] = 9
        new_grid[grid == 9] = 5
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if the grid matches the win state pattern
    # The win state has specific patterns in rows 18-29 and 42-53
    # We check if the grid matches the expected win state structure
    
    # Check rows 0-17 and 30-62 for the pattern
    for r in range(H):
        if r < 18 or r > 29 and r < 42 or r > 53:
            # These rows should have the pattern 9x36, 10x3, 9x24, 11x1
            # or 9x36, 10x1, 0x1, 10x1, 9x24, 11x1
            if r % 2 == 0:
                expected = np.array([9, 36, 10, 3, 9, 24, 11, 1])
            else:
                expected = np.array([9, 36, 10, 1, 0, 1, 10, 1, 9, 24, 11, 1])
            row = grid[r]
            # Check if the row matches the expected pattern
            if not np.array_equal(row, expected):
                return False
    
    # Check rows 18-29 and 42-53 for the pattern
    for r in range(18, 30):
        if r % 2 == 0:
            expected = np.array([9, 21, 4, 9, 9, 6, 10, 3, 9, 6, 5, 9, 9, 9, 11, 1])
        else:
            expected = np.array([9, 21, 4, 9, 9, 6, 10, 1, 0, 1, 10, 1, 9, 6, 5, 1, 9, 1, 5, 2, 9, 1, 5, 2, 9, 1, 5, 1, 9, 9, 11, 1])
        row = grid[r]
        if not np.array_equal(row, expected):
            return False
    
    for r in range(42, 54):
        if r % 2 == 0:
            expected = np.array([9, 9, 11, 9, 9, 18, 10, 3, 9, 24, 11, 1])
        else:
            expected = np.array([9, 9, 11, 9, 9, 18, 10, 1, 0, 1, 10, 1, 9, 24, 11, 1])
        row = grid[r]
        if not np.array_equal(row, expected):
            return False
    
    return True