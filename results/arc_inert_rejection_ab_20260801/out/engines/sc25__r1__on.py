import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, ACTION3 seems to move a specific pattern of colors
    # (9, 10, 2, 2) in rows 19 and 20.
    # It also changes some cells at column 62 and 63 (the far right edge).
    # The pattern [9, 10, 2, 2] is located at columns 37-40, then 35-38, etc.
    # it moves left by 2 pixels per single ACTION3 call.
    
    new_grid = grid.copy()
    if action == 3:
        # Find the target patterns in rows 19 and 20
        for r in [19, 20]:
            # Look for the sequence [9, 10, 2, 2]
            # We need to find where this pattern starts.
            # In the initial grid, it's at c=37 in row 19/20.
            # Let's search for the same pattern across the entire row.
            row = grid[r, :]
            pattern = np.array([9, 10, 2, 2])
            
            # Search for the start of the start of the pattern
            start_col = -1
            for c in range(grid.shape[1] - 4):
                if np.array_equal(row[c:c+4], pattern):
                    start_col = c
                    # break # Use only first occurrence
            
            if start_col != -1:
                # Move the pattern left by 2 columns
                # Fill old position with background color (color 5)
                new_grid[r, start_col:start_col+4] = 5
                # Place new position
                target_col = start_col - 2
                if target_col >= 0:
                    new_grid[r, target_col:target_col+4] = pattern
                else:
                    # Handle boundary
                    pass

        # Also handle the right edge changes observed in ACTION3 transitions
        # The cells at column 62 and 63 are often changed to 0.
        # The same sequence of events suggests a couple of rows are being cleared.
        # In the initial grid, r6-r16 are 14x2 at the end.
        # Thes are being set to 0.
        # Let's try to find which rows have [14, 14] at the end.
        for r in range(grid.shape[0]):
            # Find if any row ends with [14, 14]
            if np.array_equal(grid[r, -2:], [14, 14]):
                # We only need to clear two rows per action call?
                # No, the deltas show multiple pairs of rows (e.g., r6c62:0x2, r7c62:0x2).
                # la// This is part of a<|channel>thought process, but not even allowed as comments.
                pass
        
        # To match the provided deltas exactly, we would need to be a more complex state machine.
        # But based on the observed transitions, ACTION3 moves the pattern left and clears right edge cells.
        # Specifically, it looks like every time ACTION3 is called, it moves the block left by 2 columns AND
        # clears two rows of color 14 at the far right.
        
        # In the initial grid, rows 6-16 are 14x2.
        # Let's try to find which rows starting from top down that have [14, 14] and set them to 0.
        rows_to_clear = []
        for r in range(grid.shape[0]):
            if np.array_equal(grid[r, -2:], [14, 14]):
                rows_to_clear.append(r)
        
        # Clear only the first two such rows found (top-down)
        count = 0
        for r in rows_to_clear:
            if count < 2:
                new_grid[r, -2:] = 0
                count += 1
            else:
                break

    return new_grid

def is_level_complete(grid):
    # Win state not provided, but usually involves clearing all targets or reaching a goal.
    # Based on the observed transitions, we can assume completion when specific cells are cleared.
    # Check if any row ends with [14, 14] still.
    for r in range(grid.shape[0]):
        if np.array_equal(grid[r, -2:], [14, 14]):
            return False
    return True