import numpy as np

def engine(grid, action, data):
    """
    World model for ARC game re86.
    Based on observed transitions:
    ACTION4 seems to shift a vertical structure of color 9 rightwards.
    ACTION1 seems to shift structures upwards or change colors in specific patterns.
    The grid contains static walls (color 5), and moving elements (colors 4, 9, 11).
    Looking at the deltas, ACTION4 shifts a column of 9s from c39->c42->c45...
    and modifies cells around row 24.
    ACTION1 moves blocks vertically.
    """
    new_grid = grid.copy()
    h, w = new_grid.shape

    if action == 4:
        # Shift the 'column' of color 9 right by 3 units if possible
        # The observed delta shows columns 39 -> 42 -> 45
        # We look for the current position of the 9-block and move it.
        for r in range(h):
            for c in range(w - 3):
                if new_grid[r, c] == 9 and new_grid[r, c+3] == 5:
                    # This is a simplification; we need to ensure the whole block moves
                    pass
        
        # Based on the provided transitions, let's implement the specific shift seen:
        # Find where the vertical line of 9s is.
        col_of_9s = -1
        for c in range(w):
            count = np.sum(grid[:, c] == 9)
            if count > 10: # It's a long vertical line
                col_of_9s = c
                break
        
        if col_of_9s != -1 and col_of_9s + 3 < w:
            # Move column of 9s to col+3, replace old with 5 (or whatever was there)
            # But only if the target cells are color 5
            target_col = col_of_9s + 3
            for r in range(h):
                if grid[r, col_of_9s] == 9:
                    new_grid[r, target_col] = 9
                    new_grid[r, col_of_9s] = 5
            
            # Special handling for row 24 as seen in deltas
            # r24c26:5x3, r24c39:9x1, r24c42:0x1...
            # This suggests complex interaction at row 24.
            # We will approximate by shifting any non-5 values in row 24 right by 3.
            row_vals = grid[24, :].copy()
            shifted_vals = np.full(w, 5, dtype=int)
            for c in range(w - 3):
                shifted_vals[c+3] = row_vals[c]
            new_grid[24, :] = shifted_vals

    elif action == 1:
        # ACTION1 shifts things upwards.
        # Observed: r13->r10->r7->r4 (shifts of 3 rows)
        # And modifies blocks around column 10 and 21.
        
        # Shift the whole grid up by 3 if it's a "movement" phase
        # But only for specific colors? The delta shows changes in columns 10 and 21.
        # Let's implement a vertical shift for elements that aren't color 5.
        temp_grid = np.full((h, w), 5, dtype=int)
        for r in range(3, h):
            for c in range(w):
                if grid[r, c] != 5:
                    temp_grid[r-3, c] = grid[r, c]
        
        # Merge with original to keep walls (color 5)
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 5:
                    new_grid[r, c] = 5
                else:
                    new_grid[r, c] = temp_grid[r, c]

    elif action == 5:
        # ACTION5 seems to be a trigger or a small change.
        # Delta: r24c48:9x1, r27c21:0x1...
        # We will just apply the observed change if we can identify the state.
        pass

    return new_grid

def is_level_complete(grid):
    """
    The win state isn't explicitly provided as a full grid, but typically
    in these games it involves reaching a certain configuration or clearing colors.
    Given the data, we check if the 'player' block has reached the bottom or a goal.
    """
    # Check for any color 1 at the very bottom row (common ARC pattern)
    if np.any(grid[-1, :] == 1):
        return True
    # Or if all target blocks are collected/moved
    return False