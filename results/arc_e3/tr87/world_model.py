import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state based on the current grid and the given action.
    The game involves moving blocks of colors 3 and 0 horizontally across rows 48, 49, 59, and 60,
    and growing an object of color 4 from right to left along row 63.
    """
    next_grid = grid.copy()
    h, w = grid.shape

    # ACTION 4 moves specific block patterns in rows [48, 49, 59, 60] to the right by 7 units.
    if action == 4:
        rows_to_move = [48, 49, 59, 60]
        for r in rows_to_move:
            # Find the leftmost occurrence of color 3 or 0 that isn't part of the background (color 2/3)
            # In this game, these blocks start at column 15 and move right.
            # We identify the range of modified cells for each row.
            mask = (grid[r] == 3) | (grid[r] == 0)
            if np.any(mask):
                cols = np.where(mask)[0]
                c_min, c_max = cols[0], cols[-1]
                
                # Shift the identified segment to the right by 7
                segment = grid[r, c_min : c_max + 1].copy()
                new_start = c_min + 7
                new_end = c_max + 7
                
                if new_end < w:
                    next_grid[r, new_start : new_end + 1] = segment
                    # Clear old positions (set back to original background if known, here we use a default)
                    # Based on observed deltas, they are replaced by values from the shift.
                    # To be safe, we only update the shifted region.
                    # The delta shows that ACTION4 replaces specific spans.
                    # Let's simulate the run-length logic more closely.
                    pass

        # Specifically handle the growth and movement of color 4 in row 63.
        # Color 4 grows leftward every other turn. Since engine is pure, we can estimate 'turn'
        # based on the position of blocks in rows [48, 49, 59, 60].
        # Initial pos: 15. Moves: 22, 29, 36, 43...
        current_pos = 15
        mask = (grid[48] == 3) | (grid[48] == 0)
        if np.any(mask):
            current_pos = np.where(mask)[0][0]
        
        turn = (current_pos - 15) // 7
        # Growth happened at T2, T4, T6, T8, T10. These correspond to turns where a move might not have occurred.
        # However, since we don't have an explicit turn counter, let's check if it should grow.
        # In observed data, ACTION4 often triggers growth.
        c4_cols = np.where(grid[63] == 4)[0]
        if len(c4_cols) > 0:
            leftmost = c4_cols[0]
            if leftmost > 0:
                next_grid[63, leftmost - 1] = 4

    elif action == 1 or action == 2:
        # Actions 1 and 2 modify colors in the lower region (r51-r57).
        # They also trigger growth of color 4 on even transitions.
        # To simulate this without state, we use the block position as a proxy for time.
        current_pos = 15
        mask = (grid[48] == 3) | (grid[48] == 0)
        if np.any(mask):
            current_pos = np.where(mask)[0][0]
        
        turn = (current_pos - 15) // 7
        
        # Growth logic based on observation: T2, T6 were A2/A1 actions that grew C4.
        # We'll grow it if it hasn't reached its target width yet.
        c4_cols = np.where(grid[63] == 4)[0]
        if len(c4_cols) > 0:
            leftmost = c4_cols[0]
            if leftmost > 0:
                next_grid[63, leftmost - 1] = 4

    return next_grid

def is_level_complete(grid):
    """
    The level is completed when the object of color 4 in row 63 reaches a specific size/position.
    Based on observations, completion occurs after it grows to width 6 (columns 58-63).
    """
    # Check the number of pixels of color 4 in the bottom row.
    count_color_4 = np.sum(grid[63] == 4)
    # The win state was triggered when the width reached 6 and ACTION2 was performed.
    return count_color_4 >= 6