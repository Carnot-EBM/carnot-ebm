import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid is a 64x64 integer array.
    # Action 3 is the same as 'left' in many ARC games.
    # Based on observed transitions, ACTION3 moves a specific pattern of colors [9, 10, 2, 2] 
    # across columns. It also changes some cells at column 62/63 to color 0.
    # The target object is located around rows 19 and 20.
    # The target object consists of a sequence like [9, 10, 2, 2].
    # uma<|channel>thought
    # Let's refine the movement rule based on the deltas.
    # In each transition for ACTION3, the pattern [9, 10, 2, 2] shifts left by 2 units.
    # The delta shows r19c37 -> r19c35 -> r19c33 -> r19c31...
    # This suggests that when ACTION3 is pressed, an object (the block of colors) moves left.
    # Additionally, there are side effects: every few presses, two cells at col 62-63 change to 0.
    
    new_grid = grid.copy()
    
    if action == 3:
        # Find the moving object in rows 19 and 20.
        # The object is defined as the contiguous block of non-5 values in those rows.
        # We look for the first occurrence of color 9 in row 19.
        obj_rows = [19, 20]
        for r in obj_rows:
            row_data = grid[r, :]
            # Find indices where value is not 5.
            indices = np.where(row_data != 5)[0]
            if len(indices) > 0:
                start_col = indices[0]
                end_col = indices[-1]
                
                # Shift this segment left by 2 columns.
                # First, clear the old position.
                new_grid[r, start_col:end_col+1] = 5
                
                # Calculate new position.
                new_start = max(0, start_col - 2)
                new_end = max(0, end_col - 2)
                
                # Place the shifted segment.
                new_grid[r, new_start:new_end+1] = row_data[start_col:end_col+1]

        # Side effect: change cells at col 62-63 to 0.
        # The observed transitions show a pattern of changing rows (6,7), (8,9), (10,11)...
        # We need to track which rows have been changed. Since we don't have state,
        # we can check which rows in col 62 are still color 14.
        rows_to_clear = []
        for r in range(6, 64, 2):
            if grid[r, 62] == 14 and grid[r+1 < 64 and r+1 or r, 62] == 14:
                rows_to_clear.append(r)
                break # Only clear one pair per action
        
        if rows_to_clear:
            r_idx = rows_to_clear[0]
            new_grid[r_idx, 62:] = 0
            if r_idx + 1 < 64:
                new_grid[r_idx + 1, 62:] = 0

    return new_grid

def is_level_complete(grid):
    # Level complete if the moving object reaches the left edge or a specific condition.
    # Based on common ARC patterns, it might be when all cells at col 62-63 are 0.
    return np.all(grid[6:64, 62:] == 0)

import numpy as np

def is_level_complete(grid):
    """
    Check if the grid is in a win state.
    The win condition is that all cells in the grid are the same color (all 0s).
    """
    grid = np.array(grid)
    return np.all(grid == 0)
