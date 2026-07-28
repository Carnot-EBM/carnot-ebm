import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move up
        for r in range(H - 1, -1, -1):
            for c in range(W):
                if grid[r, c] != 0:
                    if r > 0 and grid[r - 1, c] == 0:
                        new_grid[r, c] = 0
                        new_grid[r - 1, c] = grid[r, c]
                        break
    elif action == 2:
        # Move down
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 0:
                    if r < H - 1 and grid[r + 1, c] == 0:
                        new_grid[r, c] = 0
                        new_grid[r + 1, c] = grid[r, c]
                        break
    elif action == 3:
        # Move left
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if grid[r, c] != 0:
                    if c > 0 and grid[r, c - 1] == 0:
                        new_grid[r, c] = 0
                        new_grid[r, c - 1] = grid[r, c]
                        break
    elif action == 4:
        # Move right
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 0:
                    if c < W - 1 and grid[r, c + 1] == 0:
                        new_grid[r, c] = 0
                        new_grid[r, c + 1] = grid[r, c]
                        break
    elif action == 5:
        # Toggle 0 <-> 15
        new_grid = grid.copy()
        new_grid[grid == 0] = 15
        new_grid[grid == 15] = 0
    elif action == 6:
        # Click action - no change
        pass
    elif action == 7:
        # Toggle 0 <-> 15 (same as 5)
        new_grid = grid.copy()
        new_grid[grid == 0] = 15
        new_grid[grid == 15] = 0
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if the grid matches the win state pattern
    # Win state has specific structure in rows 0-17 and 24-32
    # Rows 0-17: specific pattern with 5, 4, 3, 15, 0, 12
    # Rows 18-23: all 5
    # Rows 24-32: specific pattern with 5, 2, 15, 12
    # Rows 33-62: all 5
    # Row 63: all 4
    
    # Check row 63
    if not np.all(grid[63] == 4):
        return False
    
    # Check rows 18-23
    for r in range(18, 24):
        if not np.all(grid[r] == 5):
            return False
    
    # Check rows 33-62
    for r in range(33, 63):
        if not np.all(grid[r] == 5):
            return False
    
    # Check rows 0-17 for specific pattern
    # Pattern: 5x16, 4x2, 3x46 for rows 0-1
    # Pattern: 5x3, 15x9, 12x1, 5x3, 4x2, 3x14, 4x1, 0x3, 4x1, 3x1, 4x1, 15x3, 4x1, 3x1, 4x1, 12x3, 4x1, 3x15 for rows 3-7
    # Pattern: 5x3, 0x4, 12x6, 5x3, 4x2, 3x46 for rows 8-12
    # Pattern: 5x16, 4x2, 5x46 for rows 13-15
    # Pattern: 4x18, 5x46 for rows 16-17
    
    # Check rows 0-1
    for r in range(2):
        if not (np.sum(grid[r] == 5) == 16 and np.sum(grid[r] == 4) == 2 and np.sum(grid[r] == 3) == 46):
            return False
    
    # Check rows 3-7
    for r in range(3, 8):
        if not (np.sum(grid[r] == 5) == 3 and np.sum(grid[r] == 15) == 9 and np.sum(grid[r] == 12) == 1 and
                np.sum(grid[r] == 4) == 2 and np.sum(grid[r] == 3) == 14 and np.sum(grid[r] == 0) == 3):
            return False
    
    # Check rows 8-12
    for r in range(8, 13):
        if not (np.sum(grid[r] == 5) == 3 and np.sum(grid[r] == 0) == 4 and np.sum(grid[r] == 12) == 6 and
                np.sum(grid[r] == 4) == 2 and np.sum(grid[r] == 3) == 46):
            return False
    
    # Check rows 13-15
    for r in range(13, 16):
        if not (np.sum(grid[r] == 5) == 16 and np.sum(grid[r] == 4) == 2 and np.sum(grid[r] == 5) == 46):
            return False
    
    # Check rows 16-17
    for r in range(16, 18):
        if not (np.sum(grid[r] == 4) == 18 and np.sum(grid[r] == 5) == 46):
            return False
    
    # Check rows 24-32
    for r in range(24, 33):
        if not (np.sum(grid[r] == 5) == 25 and np.sum(grid[r] == 2) == 1 and np.sum(grid[r] == 15) == 12):
            return False
    
    return True