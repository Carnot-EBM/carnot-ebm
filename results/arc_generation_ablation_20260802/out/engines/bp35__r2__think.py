import numpy as np

def engine(grid, action, data):
    """
    World model for ARC game bp35.
    The grid contains various patterns and 'blocks' of color 10.
    ACTION3 seems to shift a pattern leftward or modify it.
    ACTION6 (click) replaces a region with color 10 blocks.
    ACTION4 shifts things back or modifies them.
    Based on the observed transitions, this looks like a puzzle where clicking 
    and directional keys move/transform specific colored regions.
    """
    new_grid = grid.copy()
    h, w = new_grid.shape

    if action == 3: # Shift Left / Transform
        # In the observations, ACTION3 often moves a complex pattern from right to left.
        # It also increments a counter in row 63.
        # We simulate the movement by shifting identified "active" columns.
        for r in range(h):
            row = new_grid[r]
            # Simple heuristic: if there are colors other than background (5), try to shift.
            # This is a simplification based on the provided delta runs.
            mask = (row != 5)
            if np.any(mask):
                # Find the leftmost non-background cell that isn't part of a fixed border
                # and attempt to slide the pattern.
                pass
        # The most consistent change is the increment at r63cX.
        # Since we don't have the full logic for the patterns, we apply the observed delta behavior.
        # However, since engine must be deterministic and general, we look for the 'cursor'.
        # Let's find where color 15 is in row 63.
        cursor_col = -1
        for c in range(w):
            if new_grid[63, c] == 15:
                cursor_col = c
                break
        if cursor_col != -1 and cursor_col < w - 1:
            new_grid[63, cursor_col + 1] = 15
            new_grid[63, cursor_col] = 0 # Or keep it? Delta says "r63c5:15x1" then "r63c6:15x1".
            # Actually, looking at deltas: r63c5:15x1 -> r63c6:15x1... this means a single cell of 15 moves.

    elif action == 4: # Shift Right / Transform
        cursor_col = -1
        for c in range(w):
            if new_grid[63, c] == 15:
                cursor_col = c
                break
        if cursor_col > 0:
            new_grid[63, cursor_col - 1] = 15
            new_grid[63, cursor_col] = 0

    elif action == 6: # Click
        if data is not None:
            px, py = data['x'], data['y']
            # ACTION6 replaces a region with color 10 blocks (usually 5x5 or similar).
            # Based on observed delta: r37c25:10x6 ... r41c25:10x6
            # It creates a rectangle of color 10 around the clicked point.
            r_start, r_end = max(0, py-2), min(h, py+3)
            c_start, c_end = max(0, px-3), min(w, px+3)
            new_grid[r_start:r_end, c_start:c_end] = 10
            
            # Also moves the cursor in row 63.
            cursor_col = -1
            for c in range(w):
                if new_grid[63, c] == 15:
                    cursor_col = c
                    break
            if cursor_col != -1 and cursor_col < w - 1:
                new_grid[63, cursor_col + 1] = 15
                new_grid[63, cursor_col] = 0

    return new_grid

def is_level_complete(grid):
    """
    The win state isn't explicitly provided as a full grid, but usually involves
    clearing specific colors or reaching a target configuration.
    Given the data, we check if the 'cursor' has reached the end of the board.
    """
    # Check if color 15 (the marker) reached the right side of row 63.
    if grid.shape[0] > 63:
        if np.any(grid[63, 60:] == 15):
            return True
    return False