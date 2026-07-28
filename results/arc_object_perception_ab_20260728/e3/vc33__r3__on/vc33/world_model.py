import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        if grid[py, px] == 7:
            new_grid[py, px] = 0
            # Apply gravity to the right for all rows
            for r in range(grid.shape[0]):
                row = new_grid[r, :]
                # Find all non-zero cells
                non_zero_indices = np.where(row != 0)[0]
                if len(non_zero_indices) > 0:
                    # Shift all non-zero cells to the right
                    last_idx = len(non_zero_indices) - 1
                    for i in range(len(non_zero_indices) - 1, -1, -1):
                        new_grid[r, non_zero_indices[i]] = 0
                        new_grid[r, non_zero_indices[i] + 1] = row[non_zero_indices[i]]
        return new_grid
    return grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # Win state has specific structure:
    # - Row 0 is all 7s
    # - Rows 1-15 have 0s in first 8 columns, then 3s
    # - Rows 16-19 have 9s in first 4 columns, then 0s, then 3s
    # - Rows 20-23 have 5s in first 56 columns, then 3s
    # - Rows 24-27 have 9s in first 4 columns, then 0s, then 3s
    # - Rows 28-35 have 0s in first 12 columns, then 3s
    # - Rows 36-39 have 9s in first 4 columns, then 0s, then 3s
    # - Rows 40-43 have 5s in first 28 columns, 14 2s, 5 14s, 3 20s
    # - Rows 44-47 have 9s in first 4 columns, 0s, then 3s
    # - Rows 48-51 have 0s in first 8 columns, then 3s
    # - Rows 52-53 have 0s in first 8 columns, 14 2s, 4 2s, then 3s
    # - Rows 54-55 have 0s in first 8 columns, 14 2s, 4 4s, then 3s
    # - Rows 56-57 have 0s in first 8 columns, 14 2s, 4 2s, then 3s
    # - Rows 58-63 have 0s in first 8 columns, then 3s
    
    # Check row 0
    if not np.all(grid[0, :] == 7):
        return False
    
    # Check rows 1-15
    for r in range(1, 16):
        if not np.all(grid[r, :8] == 0) or not np.all(grid[r, 8:] == 3):
            return False
    
    # Check rows 16-19
    for r in range(16, 20):
        if not np.all(grid[r, :4] == 9) or not np.all(grid[r, 4:48] == 0) or not np.all(grid[r, 48:] == 3):
            return False
    
    # Check rows 20-23
    for r in range(20, 24):
        if not np.all(grid[r, :56] == 5) or not np.all(grid[r, 56:] == 3):
            return False
    
    # Check rows 24-27
    for r in range(24, 28):
        if not np.all(grid[r, :4] == 9) or not np.all(grid[r, 4:8] == 0) or not np.all(grid[r, 8:] == 3):
            return False
    
    # Check rows 28-35
    for r in range(28, 36):
        if not np.all(grid[r, :12] == 0) or not np.all(grid[r, 12:] == 3):
            return False
    
    # Check rows 36-39
    for r in range(36, 40):
        if not np.all(grid[r, :4] == 9) or not np.all(grid[r, 4:48] == 0) or not np.all(grid[r, 48:] == 3):
            return False
    
    # Check rows 40-43
    for r in range(40, 44):
        if not np.all(grid[r, :28] == 5) or not np.all(grid[r, 28:42] == 2) or not np.all(grid[r, 42:47] == 14) or not np.all(grid[r, 47:] == 20):
            return False
    
    # Check rows 44-47
    for r in range(44, 48):
        if not np.all(grid[r, :4] == 9) or not np.all(grid[r, 4:48] == 0) or not np.all(grid[r, 48:] == 3):
            return False
    
    # Check rows 48-51
    for r in range(48, 52):
        if not np.all(grid[r, :8] == 0) or not np.all(grid[r, 8:] == 3):
            return False
    
    # Check rows 52-53
    for r in range(52, 54):
        if not np.all(grid[r, :8] == 0) or not np.all(grid[r, 8:22] == 2) or not np.all(grid[r, 22:26] == 2) or not np.all(grid[r, 26:] == 3):
            return False
    
    # Check rows 54-55
    for r in range(54, 56):
        if not np.all(grid[r, :8] == 0) or not np.all(grid[r, 8:22] == 2) or not np.all(grid[r, 22:26] == 4) or not np.all(grid[r, 26:] == 3):
            return False
    
    # Check rows 56-57
    for r in range(56, 58):
        if not np.all(grid[r, :8] == 0) or not np.all(grid[r, 8:22] == 2) or not np.all(grid[r, 22:26] == 2) or not np.all(grid[r, 26:] == 3):
            return False
    
    # Check rows 58-63
    for r in range(58, 64):
        if not np.all(grid[r, :8] == 0) or not np.all(grid[r, 8:] == 3):
            return False
    
    return True