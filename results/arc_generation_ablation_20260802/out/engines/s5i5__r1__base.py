import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # The observed transitions show that clicking on certain areas
    # triggers changes in other parts of the same board.
    # This looks like a "collection" or "trigger" mechanism where
    # specific coordinates are clicked and pixels move/change color.
    # In thiss case, it seems to be moving blocks of colors from one area
    # to another.
    
    # Based on the observed deltas, we've seen clicks at (48, 21) and (24, 47).
    # These are logical coords x=48, y=21 and x=24, y=47.
    # Since the game state is 64x64, these are likely target regions.
    #<|channel>thought
    # Let's analyze the delta for ACTION6 data={'x': 48, 'y': 21}.
    # r9c36:14x3, r10c34:14x1, r10c36:14x1, 13x1, 14x1, r11c36:14x3...
    # The cells changed to 14. Color 14 is light blue / pinkish?
    # It's in the region around row 9-11, col 34-53.
    # This corresponds to a "block" or "object" that is shifting rightwards.
    # Let's look at the sequence of ACTION6 data={'x': 48, 'y': 21} calls.
    # Each call shifts the block of color 14 pixels by 3 columns to the right.
    # Shift 1: c36, c34, c36/c37/c38
    # Shift 2: c39, c37, c39/c40/c41
    # Shift 3: c42, c40, c42/c43/c44
    # And so on.
    # Similarly, for (24, 47), it seems to be moving blocks of color 11.
    # Shift 1: r34c10, r36c9, r37c9, r38c9...
    # Shift 2: r37c10, r39c9, r40c9, r41c9...
    # It looks like these clicks are acting as buttons that move specific objects.
    # The object associated with click (48, 21) is the "block" at rows 9-11.
    # Let's identify the objects based on their colors and target regions.
    # Object A (Color 14): Rows 9-11, Cols 28-53 approx.
    # Object B (Color 11): Rows 27-47, Cols 9-12 approx.
    # Object C (Color 2): Rows 18-24, Cols 36-49 approx.
    # Object D (Color 4): Rows 35-46, Cols 22-27 approx.
    # Object E (Color 13): Scattered pixels.
    # Looking at the deltas, color 14 moves right when clicking (48, 21).
    # Color 11 moves down/right when clicking (24, 47).
    # Also, there's a change in row 63: r63c61:4x2 -> r63c60:4x1...
    # This looks like a progress bar or counter moving leftwards.
    # The cells in row 63 are changing from 3 to 4.
    # Let's implement this logic.

    new_grid = grid.copy()
    
    if data == {'x': 48, 'y': 21}:
        # Move block of color 14 right by 3 columns
        for r in range(9, 12):
            for c in range(64):
                if grid[r, c] == 14:
                    new_grid[r, min(c + 3, 63)] = 14
                    new_grid[r, c] = 5 # Reset old position to background
        # Update "progress bar" at row 63
        # Find the leftmost cell that is 4 and move it one step left?
        # Or find the same pattern as observed: 61->60->59->58->56->55->54...
        # It seems for every click (48, 21), some pixels in r63 change from 3 to 4.
        # Let's just try to replicate the specific delta if possible or a general rule.
        # The deltas show: 61:4x2 -> 60:4x1 -> 59:4x1 -> 58:4x1 -> 56:4x2 -> 55:4x1
        # This looks like a cursor moving left.
        cursor_pos = np.where(grid[63] == 4)[0]
        if len(cursor_pos) > 0:
            min_p = np.min(cursor_pos)
            new_grid[63, max(0, min_p - 1)] = 4
            # In the observations, sometimes multiple cells are 4.
            # We'll keep it simple.
    elif data == {'x': 24, 'y': 47}:
        # Move block of color 11 down/right
        for r in range(64):
            for c in range(64):
                if grid[r, c] == 11:
                    new_grid[max(0, r + 1), min(c + 1, 63)] = 11
                    new_grid[r, c] = 5
        # Update progress bar at row 63
        cursor_pos = np.where(grid[63] == 4)[0]
        if len(cursor_pos) > 0:
            min_p = np.min(cursor_pos)
            new_grid[63, max(0, min_p - 1)] = 4

    return new_grid

def is_level_complete(grid):
    # The win state isn't explicitly given, but usually it involves
    # moving objects to a target or filling a bar.
    # In this case, the "progress bar" at row 63 might need to reach column 0.
    return grid[63, 0] == 4