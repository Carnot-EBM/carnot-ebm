import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state based on observed transitions for sc25.
    Observation analysis:
    - Action 3 moves a specific pattern [9, 10, 2, 2] in rows 19 and 20 to the left.
    - Each Action 3 shift moves this pattern 2 units to the left.
    - Simultaneously, cells in the rightmost columns (62, 63) are being toggled or changed from 14 to 0.
    - The changes occur in pairs of rows (e.g., r6&r7, then r8&r9, etc.).
    """
    new_grid = grid.copy()
    
    if action == 3:
        # 1. Move the pattern [9, 10, 2, 2] in rows 19 and 20 to the left by 2.
        # First, find where color 9 is currently located in row 19.
        cols_with_9 = np.where(grid[19] == 9)[0]
        if len(cols_with_9) > 0:
            # We assume there's one primary object moving. Take the first occurrence.
            curr_col = cols_with_9[0]
            target_col = curr_col - 2
            
            # Clear old position (the 4-cell wide block)
            for r in [19, 20]:
                for c in range(curr_col, min(curr_col + 4, 64)):
                    new_grid[r, c] = 5 # Background color
            
            # Place new position
            pattern = [9, 10, 2, 2]
            for r in [19, 20]:
                for i, val in enumerate(pattern):
                    c = target_col + i
                    if 0 <= c < 64:
                        new_grid[r, c] = val

        # 2. Handle the right edge changes (columns 62 and 63).
        # The observed transitions show pairs of rows changing from 14 to 0.
        # Sequence: (6,7), (8,9), (10,11), (12,13), (14,15)...
        # We find the first pair of rows starting from row 6 that still has 14s at col 62.
        for r in range(6, 64 - 1, 2):
            if grid[r, 62] == 14:
                new_grid[r, 62] = 0
                new_grid[r, 63] = 0
                new_grid[r+1, 62] = 0
                new_grid[r+1, 63] = 0
                break
                
    return new_grid

def is_level_complete(grid):
    """
    Determines if the level is complete. 
    Usually, this happens when a target object reaches a destination or all targets are collected.
    Based on typical ARC patterns, we check if the moving block reached the far left.
    """
    cols_with_9 = np.where(grid[19] == 9)[0]
    if len(cols_with_9) > 0 and cols_with_9[0] <= 0:
        return True
    # Also check if the right edge column 14s are all gone (from row 6 onwards).
    if not np.any(grid[6:, 62:] == 14):
        return True
    return False