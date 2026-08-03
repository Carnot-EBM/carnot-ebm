import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, this looks like a puzzle game where objects move in specific directions.
    # Action 4 seems to be moving something right.
    # Action 1 seems to be moving something up.
    # Action 5 seems to be some kind of interaction or toggle.
    # Action 6 is click.
    
    new_grid = grid.copy()
    
    if action == 4: # Move Right
        # Find all cells that are not color 5 (background) and shift them right if possible.
        # We need to identify "objects" or "blocks".
        # The logic here is likely simpler: any cell of a certain value other than 5 shifts right.
        # Shift everything except background (color 5) one pixel to theright.
        # For simplicity, we
        # based on thes delta's, it's shifting columns of colors [0, 4, 9, 11]
        #<|channel>thought
        # Let' same try a general approach for movement.
        # For each row, find non-5 values and move them.
        # For each row,
        # actually, looking at the deltas, only specific blocks move.
        # The block at column 39 shifted to 42, then 45, etc.
        # 3 units per step.
        # The block at r24c48 also changed.
        # This looks like a coordinate system where objects move in steps of 3.
        
        # To implement this simply, let's just apply the observed movements.
        # Since we don't have the full state transition rules, we will use a simplified version.
        # shift all cells not equal to 5 by 3 pixels to the right if they are within bounds.
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]-1, -1, -1):
                if grid[r, c] != 5:
                    val = grid[r, c]
                    new_grid[r, c] = 5
                    if c + 3 < grid.shape[1]:
                        new_grid[r, c+3] = val
    
    elif action == 1: # Move Up
        # Shift everything except background (color 5) one pixel up.
        # Based on deltas, it moves things up by 3 rows.
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != 5:
                    val = grid[r, c]
                    new_grid[r, c] = 5
                    if r - 3 >= 0:
                        new_grid[r-3, c] = val

    elif action == 5: # Interaction/Toggle
        # Action 5 changed a few specific cells. It' same as a "collect" or "clear".
        # Let's assume it clears some objects.
        pass

    return new_grid

def is_level_complete(grid):
    # Win state usually involves clearing the board or reaching a target.
    # In this case, let's check if most of the non-background cells are gone.
    # The observed transitions don't show a win state, but we can guess.
    # return True if no non-background cells remain.
    non_bg = np.count_nonzero(grid != 5)
    return non_bg == 0