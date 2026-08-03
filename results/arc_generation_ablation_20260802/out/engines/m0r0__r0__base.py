import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where ACTION1 moves something and changes colors of blocks.
    # Based on the observed transitions, it's likely that 
    # ACTION1 shifts some state or progresses a sequence.
    # ACTION4 might be a toggle or reset.
    # Looking at the<|channel>thought process, the same cells are being modified repeatedly.
    # The coordinates correspond to columns 14-19 and 44-49.
    # These are exactly 5 pixels wide.
    # Color 10 is appearing in these regions.
    # The row ranges also shift.
    # Let's implement a simple rule based on the provided delta patterns.
    
    new_grid = grid.copy()
    
    if action == 1:
        # This action appears to move 'something' across the board or change states.
        # It often modifies rows 14-48 in cols 14-19 and 44-49.
        # We need to find the current position of color 10 blocks.
        # Find all indices of color 10.
        idx10 = np.where(grid == 10)
        if idx10[0].size > 0:
            # If there are any color 10 blocks, they seem to be moving.
            # For example, from r34-r38 (cols 14-19) to r29-r33 (cols 14-19).
            # In some cases, it moves from col 14-19 to col 44-49.
            pass

    # Since we can't induce a perfect general rule from such limited data,
    # let's try to capture the most obvious pattern: color 10 is "the player" or "the object".
    # Let's assume ACTION1 shifts the object up by 5 rows if possible.
    # Shift object (color 10) up by 5 rows.
    if action == 1:
        obj_mask = (grid == 10)
        # Move mask up by 5
        shifted_mask = np.roll(obj_mask, -5, axis=0)
        # To prevent wrap around and handle boundaries, we should actually shift.
        # We only move if the target area is not blocked by something own-colored?
        # Actually, looking at the deltas, the blocks of color 10 are moving in steps of 5.
        # The cells they leave become color 5 again.
        # The cells they enter become color 10.
        
        # Find current color 10 positions
        rows, cols = np.where(grid == 10)
        if rows.size > 0:
            # For each block of color 10, move it.
            # This is a bit complex to do generally.
            # Let' same just apply the observed delta logic for this specific level.
            pass

    # Based on the data provided, ACTION1 moves the "active" region (color 10)
    # from r34-38 -> r29-33 -> r24-28 -> r19-23 -> r14-18.
    # And then maybe shifts between columns 14-19 and 44-49.
    
    # Let's implement a simple state machine based on the sequence of actions.
    # We can use the grid itself as state.
    
    # Special case for the very first action in the trace:
    # Initial Grid has no color 10. First ACTION1 creates them at r34-38 c14-19 and r39-43 c44-49.
    # Wait, looking at INITIAL GRID again, there are some 10s already?
    # r39: 11x9, 5x5, 10x5, 5x5...  Yes, col 14-18 is color 10.
    # So rows 39-43 have color 10 at cols 14-18.
    # Rows 44-48 have color 10 at cols 44-48. (r44c44: 10x5)
    
    # Action 1: moves blocks from (39-43, 14-18) to (34-38, 14-18) AND (44-48, 44-48) to (39-43, 44-48).
    # This is a shift of -5 in row index for all color 10 blocks.
    
    if action == 1:
        mask = (grid == 10)
        new_grid[mask] = 5 # Clear old positions (assuming they return to color 5)
        # Shift mask up by 5
        shifted_rows = np.where(mask)[0] - 5
        shifted_cols = np.where(mask)[1]
        valid = (shifted_rows >= 0) & (shifted_rows < grid.shape[0])
        new_grid[shifted_rows[valid], shifted_cols[valid]] = 10
        return new_grid

    if action == 3:
        # ACTION3 changed r39-43 c44-49 from 5x5 to 5x5, 10x5.
        # It seems it "activates" or "creates" color 10 blocks at that location.
        # Let's just apply the delta if we can identify the region.
        for r in range(39, 44):
            for c in range(44, 49):
                new_grid[r, c] = 10
        return new_grid

    if action == 4:
        # ACTION4 changes rows 14-18 cols 14-19 and 44-49.
        # It looks like a toggle or shift between columns.
        # Specifically, it moves col 14-19 -> 44-49 and vice versa?
        mask_left = (grid == 10) & (np.arange(grid.shape[1]) < 20) # roughly
        mask_right = (grid == 10) & (np.arange(grid.shape[1]) >= 40)
        
        rows_l, cols_l = np.where(mask_left)
        rows_r, cols_r = np.where(mask_right)
        
        new_grid[rows_l, cols_l] = 5
        new_grid[rows_r, cols_r] = 5
        
        # Shift left to right (col + 30)
        valid_l = (cols_l + 30 < grid.shape[1])
        new_grid[rows_l[valid_l], cols_l[valid_l] + 30] = 10
        # Shift right to left (col - 30)
        valid_r = (cols_r - 30 >= 0)
        new_grid[rows_r[valid_r], cols_r[valid_r] - 30] = 10
        return new_grid

    return new_grid

def is_level_complete(grid):
    # The win state isn't explicitly given, but usually it involves moving objects to a target.
    # In this game, maybe when all color 10 blocks reach the top?
    # Or some specific configuration. Since we don't have the win state, return False.
    return False