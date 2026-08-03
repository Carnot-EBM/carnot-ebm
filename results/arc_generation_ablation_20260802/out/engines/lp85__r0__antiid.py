import numpy as np

def engine(grid, action, data):
    # The game seems to involve shifting colors/values within specific blocks (rectangles)
    # based on some trigger. ACTION0 is seen multiple times causing shifts.
    # In the observed transitions, cells at column 0 (r0-r4, r5-r9, r10-r14) are changed to color 5.
    # # This suggests a sequence of events or a state machine where each ACTION0 call rotates values.
    
    new_grid = grid.copy()
    
    if action == 0:
        # Identify all "blocks" of same-color pixels that form rectangles.
        # We're looking for the shift patterns observed in the delta.
        # Let's find the coordinates of those shifted areas.
        # Shift logic: it looks like the same set of columns [12, 18, 24, 30, 36, 42, 48]
        #
        # Based on the observed deltas, the values rotate among these positions.
        # Rotation indices for rows 19-22 and 43-46:
        # Row range 19-22:
        # Col indices: [12, 18, 24, 30, 36, 42, 48]
        # Values at these cols start as (some value), then change to others.
        # In first transition: r19c12 becomes 2, c18:10, c24:9, c30:15, c36:11, c42:2, c48:15
        # This is a rotation or permutation of existing colors.
        
        # The specific cells being changed are column 0 pixels.
        # The shift happens in blocks.
        # We umapped the laout layout.
        # Let's implement a rotation of values in those specific block coordinates.
        
        # Define the same columns that were shifted in all ACTION0 transitions.
        shift_cols = [12, 18, 24, 30, 36, 42, 48]
        
        # Rows affected by shifts
        row_groups = [
            (19, 22), # inclusive
            (25, 28),
            (31, 34),
            (37, 40),
            (43, 46)
        ]
        
        # Column 0 update (happens in groups of 5 rows)
        # Find first row where col 0 is not color 5.
        for r in range(grid.shape[0]):
            if grid[r, 0] != 5:
                new_grid[r:r+5, 0] = 5
                break
        
        # Now perform the rotations for each group.
        # For each group, we have a few different permutations.
        # The laout layout shows these are "slots" and they's rotating values.
        # Let's simulate a rotation of the same set of colors.
        #
        # In transition 1: r19c12 becomes 2, c18:10, c24:9...
        # In transition 2: r19c12 becomes 10, c18:9, c24:15...
        # In transition 3: r19c12 becomes 9, c18:15, c24:11...
        # This looks like a simple shift left/right.
        
        for rs, re in row_groups:
            for r in range(rs, re + 1):
                vals = [grid[r, c] for c in shift_cols]
                # Rotate values by 1 position to the right (or left)
                rotated_vals = vals[-1:] + vals[:-1]
                for i, c in enumerate(shift_cols):
                    new_grid[r, c] = rotated_vals[i]
                    # Since blocks are width 4, we apply it to the whole block
                    # new_grid[r, c:c+4] = rotated_vals[i]
                    # # The observed delta shows "2x4", meaning value 2 repeated 4 times.
                    # new_grid[r, c:c+4] = rotated_vals[i]
        
        # To be more precise with the same logic as deltas:
        for rs, re in row_groups:
            for r in range(rs, re + 1):
                # Get current values of the first pixel of each block
                current_vals = [grid[r, c] for c in shift_cols]
                # Shift them
                shifted_vals = np.roll(current_vals, -1) # Try shifting left
                for i, c in enumerate(shift_cols):
                    new_grid[r, c : c+4] = shifted_vals[i]

    return new_grid

def is_level_complete(grid):
    # Win state not provided, but typically involves reaching a certain configuration.
    # For now, return False unless all col 0 pixels are color 5.
    return np.all(grid[:64, 0] == 5) if grid.shape[0] >= 64 else False