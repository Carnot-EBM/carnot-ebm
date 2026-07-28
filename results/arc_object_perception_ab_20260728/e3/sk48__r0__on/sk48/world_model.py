import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 1:
        # Action 1: Move the entire grid content down by 1 row
        new_grid = np.zeros_like(grid)
        new_grid[1:, :] = grid[:-1, :]
        new_grid[0, :] = 5
        return new_grid
    elif action == 3:
        # Action 3: Move the entire grid content left by 1 column
        new_grid = np.zeros_like(grid)
        new_grid[:, 1:] = grid[:, :-1]
        new_grid[:, 0] = 5
        return new_grid
    elif action == 4:
        # Action 4: Move the entire grid content up by 1 row
        new_grid = np.zeros_like(grid)
        new_grid[:-1, :] = grid[1:, :]
        new_grid[-1, :] = 5
        return new_grid
    elif action == 6:
        # Action 6: Click action - no change
        return grid
    else:
        # Default: no change
        return grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # Win state has specific structure:
    # - Rows 0-5: all 5s
    # - Rows 6-7: 5x11, 4x42, 5x11
    # - Rows 8-28: 5x7, 2x2, 5x2, 4x42, 5x11 (with some variations in rows 25-28)
    # - Rows 29-41: 5x7, 3x2, 5x2, 4x42, 5x11 (with some variations)
    # - Rows 42-47: 5x5, 6x6, 4x42, 5x11 (with some variations)
    # - Rows 48-52: all 5s
    # - Rows 53: all 2s
    # - Rows 54-55: all 4s
    # - Rows 56-61: 4x17, 6x6, 4x41 (with some variations)
    # - Rows 62-63: all 4s
    
    # Simplified check: check if the grid matches the win state pattern
    # This is a simplified check that covers the main patterns
    
    # Check rows 0-5
    for i in range(6):
        if not np.all(grid[i, :] == 5):
            return False
    
    # Check rows 6-7
    for i in range(6, 8):
        if not (np.all(grid[i, :11] == 5) and np.all(grid[i, 11:53] == 4) and np.all(grid[i, 53:] == 5)):
            return False
    
    # Check rows 8-28
    for i in range(8, 29):
        if not (np.all(grid[i, :7] == 5) and np.all(grid[i, 7:9] == 2) and np.all(grid[i, 9:11] == 5) and np.all(grid[i, 11:53] == 4) and np.all(grid[i, 53:] == 5)):
            return False
    
    # Check rows 29-41
    for i in range(29, 42):
        if not (np.all(grid[i, :7] == 5) and np.all(grid[i, 7:10] == 3) and np.all(grid[i, 10:12] == 5) and np.all(grid[i, 12:53] == 4) and np.all(grid[i, 53:] == 5)):
            return False
    
    # Check rows 42-47
    for i in range(42, 48):
        if not (np.all(grid[i, :5] == 5) and np.all(grid[i, 5:11] == 6) and np.all(grid[i, 11:53] == 4) and np.all(grid[i, 53:] == 5)):
            return False
    
    # Check rows 48-52
    for i in range(48, 53):
        if not np.all(grid[i, :] == 5):
            return False
    
    # Check row 53
    if not np.all(grid[53, :] == 2):
        return False
    
    # Check rows 54-55
    for i in range(54, 56):
        if not np.all(grid[i, :] == 4):
            return False
    
    # Check rows 56-61
    for i in range(56, 62):
        if not (np.all(grid[i, :17] == 4) and np.all(grid[i, 17:23] == 6) and np.all(grid[i, 23:] == 4)):
            return False
    
    # Check rows 62-63
    for i in range(62, 64):
        if not np.all(grid[i, :] == 4):
            return False
    
    return True