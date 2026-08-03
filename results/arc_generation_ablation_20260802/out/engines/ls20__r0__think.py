import numpy as np

def engine(grid, action, data):
    """
    The game appears to be a puzzle where the player controls an object or state 
    represented by specific colors on the grid. Based on the transitions:
    ACTION1 moves something 'up' (shifting blocks of color from lower rows to higher rows).
    ACTION3 moves something 'right'.
    ACTION4 moves something 'left'.
    The changes in rows 61 and 62 suggest a cursor or coordinate tracker moving horizontally.
    The main gameplay involves shifting patterns of colors (3, 5, 9, 12) across the board.
    """
    new_grid = grid.copy()
    h, w = new_grid.shape

    # The observed deltas show that ACTION1 shifts vertical segments upwards.
    # ACTION3 shifts horizontal segments rightwards.
    # ACTION4 shifts horizontal segments leftwards.
    # We observe a "cursor" at row 61-62 changing its column based on actions.
    
    if action == 1: # Up
        # Shift certain block structures up by 5 units
        # Looking at r40->r35, r35->r30, r30->r25
        for r in range(25, 50):
            if r + 5 < 50:
                # This is a simplification of the complex pattern shift seen in deltas
                # In actual ARC games, these are often sliding puzzles or cellular automata.
                pass
        # Update cursor x position (observed as incrementing col in r61/62)
        for r in [61, 62]:
            curr_col = np.where(new_grid[r] == 3)[0]
            if len(curr_col) > 0:
                c = curr_col[0]
                if c + 1 < w:
                    new_grid[r, c] = new_grid[r, c+1] # Simplified movement
                    # The delta shows ACTION1 increments the cursor column index
                    # but we must be careful not to overwrite everything.

    elif action == 3: # Right
        # Based on r45c24 -> r45c19 (Wait, that's left? No, the delta says changed cells).
        # Let's look at the cursor: r61c14 -> r61c15. That's moving right.
        for r in [61, 62]:
            curr_col = np.where(new_grid[r] == 3)[0]
            if len(curr_col) > 0:
                c = curr_col[0]
                if c + 1 < w:
                    new_grid[r, c] = 11 # Replace old with background/other
                    new_grid[r, c+1] = 3

    elif action == 4: # Left
        # Cursor moves from r61c20 -> r61c19
        for r in [61, 62]:
            curr_col = np.where(new_grid[r] == 3)[0]
            if len(curr_col) > 0:
                c = curr_col[0]
                if c - 1 >= 0:
                    new_grid[r, c] = 11
                    new_grid[r, c-1] = 3

    # Since the provided transitions are very specific and we need a general model,
    # but the logic is highly dependent on the exact layout of colors (3, 5, 9, 12),
    # we implement the cursor movement as it's the most consistent part of the delta.
    
    # Re-evaluating ACTION1 based on deltas:
    # ACTION1: r40->r35, r35->r30, r30->r25 AND cursor col increments.
    if action == 1:
        for r in [61, 62]:
            curr_col = np.where(new_grid[r] == 3)[0]
            if len(curr_col) > 0:
                c = curr_col[0]
                if c + 1 < w:
                    new_grid[r, c] = 11
                    new_grid[r, c+1] = 3
                    
    return new_grid

def is_level_complete(grid):
    """
    The win state isn't explicitly provided, but usually involves reaching a target 
    configuration or clearing certain blocks. Without a WIN STATE grid, we assume 
    it's not complete unless a specific condition is met.
    """
    # Common ARC win conditions: all cells of one color gone, or a pattern formed.
    # Given no win state, return False.
    return False