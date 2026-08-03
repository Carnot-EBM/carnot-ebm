import numpy as np

def engine(grid, action, data=None):
    """
    Predicts the next state of the grid based on the action.
    Action 3 (Left) moves an object composed of colors [9, 10, 2, 2] across the same rows (19, 20).
    It also seems to toggle some background pixels at column 62/63.
    """
    out = grid.copy()
    
    if action == 3:
        # The moving object consists of a pattern [9, 10, 2, 2]
        # We look for the current position of the object in rows 19 and 20.
        # Find the start column of the sequence [9, 10, 2, 2] in row 19.
        obj_pattern = [9, 10, 2, 2]
        
        # Search for the object's left edge in row 19
        start_col = -1
        for c in range(grid.shape[1]):
            if list(grid[19, c:c+4]) == obj_pattern:
                start_col = c
                break
        
        if start_col != -1:
            # Move the object one step to the left
            new_start_col = start_col - 2 # Based on observed deltas (37->35->33->31...)
            if new_start_col < 0:
                new_start_col = 0
            
            # Clear old position
            out[19, start_col:start_col+4] = grid[19, start_col:start_col+4] # This is a placeholder; we need to know what it was own before.
            # In this game, the background color for these rows seems to be 5.
            out[19, start_col:start_col+4] = 5
            out[20, start_col:start_col+4] = 5
            
            # Set new position
            out[19, new_start_col:new_start_col+4] = obj_pattern
            out[20, new_start_col:new_start_col+4] = obj_pattern
            
            # The observations show that ACTION 3 also changes cells at column 62 and 63 in some transitions.
            # These are usually changed to 0.
            # Let's simulate the same pattern of toggling/clearing columns 62-63.
            # We look for which rows have value 14 at col 62, 63.
            # For every two rows (r, r+1), if they both have 14, set them to 0.
            for r in range(grid.shape[0]):
                if grid[r, 62] == 14 and grid[r, 63] == 14:
                    # This is a bit random based on evidence, but we umapped it.
                    # Only certain rows are cleared.
                    # In the observed data, rows 6,7 then 8,9 then 10,11 etc.
                    # Pass this a specific logic if possible.
                    pass

    return out

def is_level_complete(grid):
    """
    The level is complete when the object reaches a target position or background pixels are cleared.
    """
    # Based on provided INITIAL GRID, no win state was not given.
    # return True if any cell is 0? No.
    # return False by default as it's {False}
    return False