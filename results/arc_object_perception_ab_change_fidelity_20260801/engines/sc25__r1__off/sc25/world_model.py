import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, ACTION3 seems to move a specific pattern of colors (9, 10, 2, 2) 
    # across the same rows (19 and 20), shifting it leftwards by 2 columns each time.
    # It also toggles some values in column 62 and 63 (the last two columns).
    #
    # Inducing general rules:
    # - The "player" or "active object" is likely the pattern [9, 10, 2, 2] in rows 19 and 20.
    # - ACTION3 moves this object leftward.
    # - Column 62-63 acts as a trigger or state indicator.
    #
    # Let's implement the movement logic for the active object.
    
    new_grid = grid.copy()
    
    if action == 3:
        # Find the current position of the active object (pattern [9, 10, 2, 2])
        # We look for the first occurrence of color 9 in row 19.
        # start_col = np.where(grid[19] == 9)[0][0] if any(grid[19] == 9) else None
        
        # In the observed data, the same block of colors shifts left.
        # Shift the block [9, 10, 2, 2] in rows 19 and 20.
        # find the column where the value 9 starts the block
        for r in [19, 20]:
            # Search for the pattern [9, 10, 2, 2]
            for c in range(62):
                if (grid[r, c] == 9 and grid[r, c+1] == 10 and 
                    grid[r, c+2] == 2 and grid[r, c+3] == 2):
                    # Replace old position with background (color 5)
                    new_grid[r, c:c+4] = 5
                    # Place new position shifted left by 2 columns
                    if c - 2 >= 0:
                        new_grid[r, c-2:c+2] = [9, 10, 2, 2]
                    else:
                        # Handle boundary
                        pass
    
    # The transitions also show changes to the last two columns (62, 63).
    # These are often color 14 or 0.
    # ACTION3 triggers a change in rows 6, 7, 8... etc.
    # In thes observed data, it's specifically changing some cells to 0.
    # We can't fully induce the exact trigger logic for column 62/63 without more frames,
    # but we can actually see that the laast two columns are 14x2 which means colors 62 and 63 are both 14.
    # Since the same pattern moves left, let's assume the action is move left.
    # 
    # # Let's<|channel>thought
    # # Correcting based on the delta:
    # # r19c37:9x1,10x1,2x2 -> this means at row 19, col 37, values are now 9, 10, 2, 2.
    # # r20c37:9x1,10x1,2x2 -> same for row 20.
    # # r6c62:0x2 -> row 6, col 62 becomes 0 (for 2 cells).
    # # r7c62:0x2 -> row 7, col 62 becomes 0 (for 2 cells).
    # # etc.
    # # a simple shift of the block [9, 10, 2, 2] in rows 19-20.
    # # a sequence of pairs of rows in column 62/63 being set to 0.
    
    # To be precise with the observed deltas:
    # ACTION3 shifts the pattern [9, 10, 2, 2] left by 2 columns each time.
    # It also sets two rows' last two columns to 0 starting from row 6.
    
    # Let's implement this specific behavior.
    
    # Find current position of color 9 in row 19.
    try:
        curr_col = np.where(grid[19] == 9)[0][0]
        new_col = curr_col - 2
        if new_col >= 0:
            # Clear old and place new
            new_grid[19, curr_col:curr_col+4] = 5
            new_grid[20, curr_col:curr_col+4] = 5
            new_grid[19, new_col:new_col+4] = [9, 10, 2, 2]
            new_grid[20, new_col:new_col+4] = [9, 10, 2, 2]
    except IndexError:
        pass

    # Handle column 62/63 (the "progress" indicator)
    # Count how many times ACTION3 has been called by counting zeros in col 62.
    zeros_count = np.sum(grid[:, 62] == 0)
    # The observed data shows rows 6,7 then 8,9 then 10,11... being set to 0.
    # Each action sets two more rows to 0.
    start_row = 6 + (zeros_count // 2) * 2
    if start_row < 64:
        new_grid[start_row : start_row + 2, 62:] = 0
        
    return new_grid

def is_level_complete(grid):
    # Win state usually occurs when the object reaches a target or progress bar is full.
    # In this case, let's assume it's complete if the pattern [9, 10, 2, 2] reaches the left edge.
    try:
        curr_col = np.where(grid[19] == 9)[0][0]
        return curr_col <= 0
    except IndexError:
        return False