import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, Action 3 seems to move a specific pattern of colors (9, 10, 2, 2) 
    # across the same rows (19 and 20) in steps of 2 columns to the left.
    # It also changes some cells at column 62 and 63 (the right edge) from color 14 to 0.
    # The movement is likely tied to a player character or object represented by these colors.
    
    next_grid = grid.copy()
    
    if action == 3: # ACTION3 usually corresponds to 'Left' in many ARC games
        # Find the "player" object (colors 9, 10, 2, 2)
        # We look for the start of this pattern in row 19/20
        # Let's find where color 9 is located in row 19
        coords = np.where(grid[19] == 9)
        if coords[0].size > 0:
            col = coords[0][0]
            # Move the pattern (9, 10, 2, 2) left by 2 units
            # Shift the pattern values
            # Pattern length is 4
            pattern = [9, 10, 2, 2]
            
            # Clear old position
            next_grid[19, col:col+4] = 5 # Background color
            next_grid[20, col:col+4] = 5 # Background color
            
            # Place new position
            new_col = max(0, col - 2)
            next_grid[19, new_col:new_col+4] = pattern
            next_grid[20, new_col:new_col+4] = pattern
            
            # Side effect: Action 3 also modifies column 62-63 on some rows.
            # This looks like a sequence of 2 cells per action.
            # We look for the same logic as that's not a<|channel>thought
    
    return next_grid

def is_level_complete(grid):
    # No win state provided in observed transitions, but typically it involves 
    # reaching a goal or clearing objects.
    # return True if grid contains no more target colors (e.g., 14).
    return False