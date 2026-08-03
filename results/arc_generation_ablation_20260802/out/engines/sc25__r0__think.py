import numpy as np

def engine(grid, action, data):
    """
    World model for sc25.
    Based on observed transitions:
    ACTION3 seems to move a specific pattern (9x1, 10x1, 2x2) leftwards in rows 19 and 20.
    Simultaneously, it clears blocks of color 14 at the right edge (col 62, 63).
    The movement happens in steps of 2 columns per single ACTION3 call.
    The clearing of col 62/63 happens in pairs of rows.
    """
    new_grid = grid.copy()
    
    if action == 3:
        # Pattern moving left in rows 19 and 20
        # The pattern is [9, 10, 2, 2]
        pattern = [9, 10, 2, 2]
        for r in [19, 20]:
            # Find current position of the pattern start (color 9)
            # We look for the first occurrence of 9 that is followed by 10
            cols = np.where(new_grid[r] == 9)[0]
            for c in cols:
                if c + 1 < new_grid.shape[1] and new_grid[r, c+1] == 10:
                    # Move pattern left by 2 units if possible
                    start_col = c
                    end_col = c + len(pattern) - 1
                    
                    # Clear old position
                    new_grid[r, start_col : end_col + 1] = 5 # Background color
                    
                    # Place at new position
                    new_pos = max(0, start_col - 2)
                    for i, val in enumerate(pattern):
                        if new_pos + i < new_grid.shape[1]:
                            new_grid[r, new_pos + i] = val
                    break

        # Clearing logic for col 62-63
        # The observed transitions show pairs of rows being cleared from 14 to 0.
        # This looks like a sequence: (6,7), (8,9), (10,11)...
        # We need to track state or derive it. Since we don't have state, 
        # we look for the first pair of rows starting from row 6 that still has 14s.
        for r in range(6, 64, 2):
            if new_grid[r, 62] == 14 or new_grid[r+1, 62] == 14:
                new_grid[r, 62:64] = 0
                new_grid[r+1, 62:64] = 0
                break
                
    return new_grid

def is_level_complete(grid):
    """
    The level is complete when the target pattern reaches its destination 
    or all obstacles are cleared. Based on typical ARC patterns, 
    we check if the right column is fully cleared or the object reached x=0.
    """
    # Check if col 62/63 are mostly 0s now
    right_col = grid[:, 62:]
    if np.all(right_col == 0) or np.sum(right_col == 14) < 5:
        return True
    return False