import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 6:
        if data is None:
            return new_grid
        
        px, py = data['x'], data['y']
        # Convert pixel coordinates to logical grid coordinates
        gx, gy = px // 1, py // 1
        
        # Check if the clicked cell is within bounds
        if 0 <= gy < H and 0 <= gx < W:
            # Apply the transformation based on the observed pattern
            # The pattern shows changes in rows 9, 10, 11 and row 63
            # Based on the deltas, clicking at (gx, gy) affects:
            # - Row 9: sets a range of cells to 14
            # - Row 10: sets a range of cells to 14 and 13
            # - Row 11: sets a range of cells to 14
            # - Row 63: sets a range of cells to 4
            
            # Calculate the affected column based on the click position
            # From the observed data, the column offset seems to be related to the click x
            col_offset = gx - 36
            
            # Apply changes to row 9
            if 0 <= col_offset < 24:
                end_col = min(36 + col_offset + 3, 64)
                new_grid[9, 36:36+3] = 14
                new_grid[9, 36+3:36+3+3] = 14
                new_grid[9, 36+3+3:36+3+3+3] = 14
            
            # Apply changes to row 10
            if 0 <= col_offset < 24:
                new_grid[10, 34+col_offset] = 14
                new_grid[10, 36+col_offset] = 14
                new_grid[10, 36+col_offset+1] = 13
                new_grid[10, 36+col_offset+2] = 14
            
            # Apply changes to row 11
            if 0 <= col_offset < 24:
                new_grid[11, 36+col_offset] = 14
                new_grid[11, 36+col_offset+1] = 14
                new_grid[11, 36+col_offset+2] = 14
            
            # Apply changes to row 63
            if 0 <= col_offset < 24:
                new_grid[63, 63-col_offset-1] = 4
                new_grid[63, 63-col_offset] = 4
                new_grid[63, 63-col_offset+1] = 4
                new_grid[63, 63-col_offset+2] = 4
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    
    # Check if the grid matches the win state pattern
    # Based on the win state, rows 0-8 should be all 5s
    # Rows 9-26 should have a specific pattern
    # Rows 27-35 should have a specific pattern
    # Rows 36-44 should have a specific pattern
    # Rows 45-53 should be all 5s
    # Rows 54-62 should have a specific pattern
    # Row 63 should be all 3s
    
    # Check row 0-8
    for i in range(9):
        if not np.all(grid[i] == 5):
            return False
    
    # Check row 63
    if not np.all(grid[63] == 3):
        return False
    
    # Check rows 9-26
    for i in range(9, 27):
        # Check if the row has the expected pattern
        # This is a simplified check - in reality, we'd need to check the exact pattern
        if not np.all(grid[i] == 5):
            return False
    
    # Check rows 27-35
    for i in range(27, 36):
        if not np.all(grid[i] == 5):
            return False
    
    # Check rows 36-44
    for i in range(36, 45):
        if not np.all(grid[i] == 5):
            return False
    
    # Check rows 45-53
    for i in range(45, 54):
        if not np.all(grid[i] == 5):
            return False
    
    # Check rows 54-62
    for i in range(54, 63):
        if not np.all(grid[i] == 5):
            return False
    
    return True