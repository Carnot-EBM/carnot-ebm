import numpy as np

def engine(grid, action, data):
    if action == 6:
        return grid
    
    if action == 1:
        return apply_action_1(grid)
    elif action == 2:
        return apply_action_2(grid)
    elif action == 3:
        return apply_action_3(grid)
    elif action == 4:
        return apply_action_4(grid)
    else:
        return grid

def apply_action_1(grid):
    h, w = grid.shape
    new_grid = grid.copy()
    
    # Action 1 toggles specific columns in rows 19-22
    # Based on observed changes:
    # r19c37:8x2, r20c37:8x2, r21c35:4x2, r22c35:4x2
    # This suggests toggling columns 37 and 35 in rows 19-22
    
    # Toggle column 37 in rows 19-22
    for r in range(19, 23):
        if r < h and c < w:
            new_grid[r, c] = 1 - new_grid[r, c]
    
    # Toggle column 35 in rows 21-22
    for r in range(21, 23):
        if r < h and c < w:
            new_grid[r, c] = 1 - new_grid[r, c]
    
    return new_grid

def apply_action_2(grid):
    h, w = grid.shape
    new_grid = grid.copy()
    
    # Action 2 toggles specific columns in rows 19-22
    # Based on observed changes:
    # r19c35:4x4, r20c35:4x4, r21c35:8x4, r22c35:8x4
    # This suggests toggling column 35 in rows 19-22
    
    # Toggle column 35 in rows 19-22
    for r in range(19, 23):
        if r < h and c < w:
            new_grid[r, c] = 1 - new_grid[r, c]
    
    return new_grid

def apply_action_3(grid):
    h, w = grid.shape
    new_grid = grid.copy()
    
    # Action 3 toggles specific columns in rows 19-22
    # Based on observed changes:
    # r19c35:8x2,4x2,5x4, r20c35:8x2,4x2,5x4, r21c35:8x2,4x2,5x4, r22c35:8x2,4x2,5x4
    # This suggests toggling columns 35, 37, 39 in rows 19-22
    
    # Toggle column 35 in rows 19-22
    for r in range(19, 23):
        if r < h and c < w:
            new_grid[r, c] = 1 - new_grid[r, c]
    
    # Toggle column 37 in rows 19-22
    for r in range(19, 23):
        if r < h and c < w:
            new_grid[r, c] = 1 - new_grid[r, c]
    
    # Toggle column 39 in rows 19-22
    for r in range(19, 23):
        if r < h and c < w:
            new_grid[r, c] = 1 - new_grid[r, c]
    
    return new_grid

def apply_action_4(grid):
    h, w = grid.shape
    new_grid = grid.copy()
    
    # Action 4 toggles specific columns in rows 19-22
    # Based on observed changes:
    # r19c39:4x2,8x2, r20c39:4x2,8x2, r21c39:4x2,8x2, r22c39:4x2,8x2
    # This suggests toggling columns 39 and 41 in rows 19-22
    
    # Toggle column 39 in rows 19-22
    for r in range(19, 23):
        if r < h and c < w:
            new_grid[r, c] = 1 - new_grid[r, c]
    
    # Toggle column 41 in rows 19-22
    for r in range(19, 22):
        if r < h and c < w:
            new_grid[r, c] = 1 - new_grid[r, c]
    
    return new_grid

def is_level_complete(grid):
    h, w = grid.shape
    
    # Check if the grid matches the win state pattern
    # Based on the initial grid and transitions, the win state has specific patterns
    # Check for the presence of specific color patterns
    
    # Check if all rows have the expected pattern
    for r in range(h):
        row = grid[r]
        # Check if the row matches the expected pattern
        # This is a simplified check - in reality, we'd need to know the exact win state
        
        # Check for the presence of specific colors
        if not np.any(row == 14) or not np.any(row == 13):
            return False
    
    return True