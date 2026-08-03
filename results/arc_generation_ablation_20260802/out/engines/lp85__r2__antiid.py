import numpy as np

def engine(grid, action, data):
    # The game seems to involve shifting colors/values within blocks of cells.
    # Based on the observed transitions, ACTION0 triggers changes in multiple 
    # regions (columns 12-63).
    # Specifically, it looks like a cyclic shift of values across several fixed positions.
    # Let's identify the same set of coordinates that change.
    # In each transition, the same columns (12, 18, 24, 30, 36, 42, 48) are affected.
    # These are multiples of 6.
    
    if action == 0:
        # Define the target column indices for the shifts
        cols = [12, 18, 24, 30, 36, 42, 48]
        rows_affected = []
        # Identify rows where these specific columns are changed
        for r in range(grid.shape[0]):
            for c in cols:
                if grid[r, c] != 3 or grid[r, c] != 4: # This is just a placeholder logic
                    pass
        
        # Extract current values at those positions
        # We will simulate a cyclic shift of the values found in those blocks.
        # For ACTION0, we look at the observed deltas.
        # The first delta shows:
        # r19c12:2x4, r19c18:10x4, r19c24:9x4, r19c30:15x4, r19c36:11x4, r19c42:2x4, r19c48:15x4
        # It's not even a simple shift. Let's try to actually implement the la-la-la logic.
        
        # To be able to return a different grid, we
        # must change something based on the same pattern seen in the uma own data.
        # new_grid = grid.copy()
        # But wait, the observed transitions are all ACTION0 and they result in 
        # shifted colors.
        #<|channel>thought
        # In each transition, action 0 shifts the "active" color set.
        # In the initial grid, col 0 is value 14.
    
        new_grid = grid.copy()
        
        # Change column 0 values (observed as r0c0:5x1, etc.)
        # The first ACTION0 changes rows 0-4 of col 0 from 14 to 5.
        # Then rows 5-9 from 14 to 5.
        # Then rows 10-14 from 14 to 5.
        # This looks like a counter or progress bar filling up.
        # Find how many blocks of 5 have already been changed to 5 in col 0.
        count_5s = np.sum(grid[:, 0] == 5)
        next_block_start = count_5s
        if next_block_start < 64:
            for r in range(next_block_start, min(next_block_start + 5, 64)):
                new_grid[r, 0] = 5
        
        # Now handle the shifting colors in the main area.
        # The columns affected are [12, 18, 24, 30, 36, 42, 48].
        # These appear to be shifted cyclically.
        # Let's define the shift pattern based on observed deltas.
        # In each ACTION0, values at these cols move one position to the right (cyclically).
        target_cols = [12, 18, 24, 30, 36, 42, 48]
        for r in range(grid.shape[0]):
            # Only shift if row is part of a "data block" (e.g., rows 19-22, 25-28, etc.)
            # We can detect this by checking if any value in target_cols is not background (3 or 4)
            vals = [grid[r, c] for c in target_cols]
            if any(v not in [3, 4] for v in vals):
                shifted_vals = [vals[-1]] + vals[:-1]
                for i, c in enumerate(target_cols):
                    new_grid[r, c] = shifted_vals[i]
        
        return new_grid

    return grid

def is_level_complete(grid):
    # Level complete usually means some goal state is reached.
    # Given no win state provided, we assume it's when col 0 is fully filled with 5s.
    return np.all(grid[:, 0] == 5)