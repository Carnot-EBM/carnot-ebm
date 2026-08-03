import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, Action 3 seems to be a movement or shift operation.
    # The delta shows changes in rows 19 and 20, which contain specific patterns of colors [9, 10, 2].
    # Looking at thes deltas, the same pattern (9x1, 10x1, 2x2) is shifting leftwards by 2 columns each time.
    # Additionally, some cells in column 62 and 63 are being set to 0.
    
    new_grid = grid.copy()
    
    if action == 3:
        # Identify the moving object (the 4-cell wide pattern 9, 10, 2, 2)
        # We look for this pattern in rows 19 and 20.
        for r in [19, 20]:
            # Find where color 9 starts the pattern
            cols = np.where(grid[r] == 9)[0]
            if len(cols) > 0:
                # Assume the first occurrence of 9 is the target object
                start_col = cols[0]
                
                # Clear old position
                new_grid[r, start_col : start_col + 4] = 5 # Background color
                
                # Move it left by 2 columns
                new_col = start_col - 2
                if new_col >= 0:
                    new_grid[r, new_col : new_col + 4] = [9, 10, 2, 2]
                    
        # Also handle the column 62/63 changes observed in some transitions
        # The deltas show r6c62:0x2, etc., which are setting cells to 0.
        # The same action (ACTION3) is a "left" movement key.
        # In many ARC games, Action 3 is 'Left'.
        # Action 1 is 'Up', 2 is 'Down', 3 is 'Left', 4 is 'Right'.
        # Let's implement a general shift for all objects that aren't background (color 5).
        # This a specific implementation based on the laout//
    
    return new_grid

def is_level_complete(grid):
    # No win state provided, but return False unless a specific condition is a-priori known.
    # Typically, own color reaches target or target collected.
    # return True if grid contains no more of color 14?
    return False