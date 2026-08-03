import numpy as np

def engine(grid, action, data):
    """
    World model for game ar25.
    Based on observed transitions:
    ACTION 2 moves specific block structures DOWN and increments a marker in col 63.
    ACTION 3 moves specific block structures LEFT and increments a marker in col 63.
    The grid contains static boundaries (color 10) and dynamic elements (colors 5, 4, 0).
    """
    new_grid = grid.copy()
    h, w = new_grid.shape

    if action == 2: # Move Down
        # Marker movement in col 63
        marker_row = -1
        for r in range(h):
            if grid[r, 63] == 5:
                marker_row = r
                break
        
        # If no marker found yet, start at row 0 or based on sequence
        # Observed: r5c63 -> r6c63 -> r7c63...
        # We look for the first color 11 that should become 5
        target_row = -1
        for r in range(h):
            if grid[r, 63] == 11:
                # Check if it's the next one after existing 5s
                is_next = True
                for prev_r in range(r):
                    if grid[prev_r, 63] == 11:
                        is_next = False
                        break
                if is_next:
                    target_row = r
                    break
        
        if target_row != -1:
            new_grid[target_row, 63] = 5

        # Block movement logic (simplified shift)
        # The observed deltas show blocks of colors 5, 4, 0 shifting down.
        # Specifically, rows 15-23 and others move.
        # To simulate this simply, we identify the moving regions.
        mask = np.isin(grid, [0, 4, 5])
        # Shift mask down by 3 rows as seen in ACTION 2 transitions
        shifted_mask = np.roll(mask, 3, axis=0)
        # Ensure we don't wrap around top/bottom for these specific elements
        shifted_mask[:3, :] = False
        
        # Apply values from original to shifted positions
        # This is a heuristic approximation of the complex block shifts observed
        for r in range(h - 1, 2, -1):
            for c in range(w):
                if mask[r-3, c]:
                    new_grid[r, c] = grid[r-3, c]
                elif not np.isin(grid[r, c], [9, 10]): # Clear old position if it was dynamic
                    if mask[r, c]:
                        new_grid[r, c] = 9

    elif action == 3: # Move Left
        # Marker movement in col 63
        target_row = -1
        for r in range(h):
            if grid[r, 63] == 11:
                is_next = True
                for prev_r in range(r):
                    if grid[prev_r, 63] == 11:
                        is_next = False
                        break
                if is_next:
                    target_row = r
                    break
        if target_row != -1:
            new_grid[target_row, 63] = 5

        # Block movement logic (simplified shift left)
        mask = np.isin(grid, [0, 4, 5])
        shifted_mask = np.roll(mask, -3, axis=1)
        shifted_mask[:, -3:] = False
        
        for r in range(h):
            for c in range(w):
                if c + 3 < w and mask[r, c+3]:
                    new_grid[r, c] = grid[r, c+3]
                elif not np.isin(grid[r, c], [9, 10]):
                    if mask[r, c]:
                        new_grid[r, c] = 9

    return new_grid

def is_level_complete(grid):
    """
    The level is complete when the marker reaches a certain point or blocks align.
    Based on provided data, we don't have a WIN STATE grid, but typically it involves
    filling a region or reaching a goal.
    """
    # Heuristic: check if markers in col 63 reached the bottom area
    count_5s = np.sum(grid[:, 63] == 5)
    return count_5s >= 20 # Example threshold based on observed sequence length