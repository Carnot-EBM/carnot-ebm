import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        # Action 6 is a click at (px, py)
        # Based on observed transitions, clicking a cell toggles it to color 5
        # and also toggles a symmetric cell at (63-py, 63-px) to color 5
        grid[py, px] = 5
        grid[63-py, 63-px] = 5
        return grid
    return grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # Win state has specific structure:
    # - Row 63 is all 0s
    # - Rows 0-9 have specific patterns
    # - Rows 10-22 are all 5s
    # - Rows 23-31 have specific patterns
    # - Rows 32-36 are all 5s
    # - Rows 37-41 have specific patterns
    # - Rows 42-57 are all 5s
    # - Rows 58-62 are all 5s
    
    # Check row 63 is all 0s
    if not np.all(grid[63, :] == 0):
        return False
    
    # Check rows 0-9 have specific patterns
    for i in range(10):
        if not np.all(grid[i, :] == 5) and not np.all(grid[i, :] == 4):
            return False
    
    # Check rows 10-22 are all 5s
    for i in range(10, 23):
        if not np.all(grid[i, :] == 5):
            return False
    
    # Check rows 23-31 have specific patterns
    for i in range(23, 32):
        if not np.all(grid[i, :] == 5) and not np.all(grid[i, :] == 9):
            return False
    
    # Check rows 32-36 are all 5s
    for i in range(32, 37):
        if not np.all(grid[i, :] == 5):
            return False
    
    # Check rows 37-41 have specific patterns
    for i in range(37, 42):
        if not np.all(grid[i, :] == 5) and not np.all(grid[i, :] == 10):
            return False
    
    # Check rows 42-57 are all 5s
    for i in range(42, 58):
        if not np.all(grid[i, :] == 5):
            return False
    
    # Check rows 58-62 are all 5s
    for i in range(58, 63):
        if not np.all(grid[i, :] == 5):
            return False
    
    return True