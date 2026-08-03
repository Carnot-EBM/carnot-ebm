import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, ACTION3 seems to move a specific pattern of colors (9, 10, 2, 2)
    # and potentially change some boundary values (color 14 -> 0).
    # The pattern [9, 10, 2, 2] appears in rows 19 and 20.
    # In each transition, this pattern moves left by 2 columns per single ACTION3 call.
    # Move the pattern [9, 10, 2, 2] in rows 19 and 20.
    new_grid = grid.copy()
    
    if action == 3:
        # Find the current position of the pattern [9, 10, 2, 2] in rows 19 and 20.
        # We are looking for the start of the sequence 9, 10, 2, 2 in row 19.
        for r in [19, 20]:
            row = new_grid[r]
            # Search for the first occurrence of color 9 followed by 10, 2, 2.
            # For c in range(64):
            #     if row[c] == 9 and row[c+1] == 10 and row[c+2] == 2 and row[c+3] == 2:
            #     # Found it.
            #     # a bit simplified search for thes specific colors
            pass

        # Based on the observed deltas, the pattern moves left by 2 columns.
        # The pattern is at (19, 37), then (19, 35), then (19, 33), etc.
        # In some transitions, therethought they also change cells at column 62-63.
        # 
        # Let's try to find the current position of the pattern and shift it.
        # Find the starting column 'c' where grid[19, c] == 9 and grid[19, c+1] == 10.
        start_col = -1
        for c in range(64):
            if grid[19, c] == 9 and grid[19, c+1] == 10:
                start_col = c
                break
        
        if start_col != -1:
            # Shift the pattern [9, 10, 2, 2] left by 2 units.
            # Restore old position to background color 5.
            new_grid[19, start_col : start_col + 4] = 5
            new_grid[20, start_col : start_col + 4] = 5
            
            # Place new position.
            new_col = start_col - 2
            if new_col >= 0:
                new_grid[19, new_col : new_col + 4] = [9, 10, 2, 2]
                new_grid[20, new_col : new_col + 4] = [9, 10, 2, 2]
            else:
                # The pattern might wrap or stop.
                pass

        # Also handle the boundary changes (color 14 -> 0).
        # In some ACTION3 calls, the same cells are changed to 0.
        # For example, r6c62:0x2 means grid[6, 62]=0 and grid[6, 63]=0.
        # For each transition, two rows of column 62-63 change from 14 to 0.
        # These rows are processed in a sequence: (6,7), (8,9), (10,11), etc.
        # We're looking for the first pair of rows where grid[r, 62:64] == [14, 14].
        for r in range(64):
            if new_grid[r, 62] == 14 and new_grid[r, 63] == 14:
                # Change them to 0.
                # Only do this for one pair of rows per action call?
                # Let's check the observed transitions.
                # 1st: no boundary change.
                # 2nd: r6, r7.
                # 3rd: r8, r9.
                # 5th: r10, r11.
                # 6th: r12, r13.
                # 8th: r14, r15.
                # This is roughly every other ACTION3 call or some specific trigger.
                # However, let's try to find the same pattern of removal.
                pass

    return new_grid

def is_level_complete(grid):
    # Win state not provided, but usually it's when a target object is moved to a goal or all targets are removed.
    # return True if grid[19, 0] == 9 # Example condition.
    return False