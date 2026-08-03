import numpy as np

def engine(grid, action, data):
    # Grid shape 64x64
    # Action 4: Right movement of some object (likely a column/block)
    # Action 1: Upward movement of some object (likely a column/block)
    # Action 5: Some toggle or interaction
    # Action 6: Click (not only used here but potentially as a<|channel>thought process)
    
    new_grid = grid.copy()
    
    if action == 4:
        # Observed ACTION4 shifts a vertical structure at col 39->42->45...
        # It also affects cells around r24 and r63.
        # Let's find the 'object' being moved.
        # The observed changes suggest a shift to the right by 3 columns.
        # We identify the region of interest based on the delta.
        # In ACTION4, we see r11-r37 shifting from c39 to c42 etc.
        # For simplicity, we implement a general "shift right" for specific colors.
        # shifted_cols = [39, 42, 45]
        # Looking at the deltas, it seems like color 9 is moving.
        # Color 9 is often associated with the moving parts in these puzzles.
        # Find all pixels of color 9 and move them right by 3.
        mask = (grid == 9)
        coords = np.argwhere(mask)
        for r, c in coords:
            new_grid[r, c] = 5 # Reset old position to background (color 5)
            if c + 3 < 64:
                new_grid[r, c+3] = 9
        # Special case for r24 and r63 as seen in deltas
        # This looks like a counter or state tracker at r63c58...54.
        # The ACTION4 deltas show r63c58 -> r63c57 -> r63c56.
        # It's a decrementing index/counter on the bottom row.
        # We need to find where the '1' is on row 63.
        # For action 4, we shift the '1' left by 1 on row 63.
        row63 = grid[63, :]
        one_pos = np.where(row63 == 1)[0]
        if len(one_pos) > 0:
            old_pos = one_pos[0]
            new_grid[63, old_pos] = 15
            if old_pos - 1 >= 0:
                new_grid[63, old_pos - 1] = 1
    
    elif action == 1:
        # Observed ACTION1 shifts structures upward.
        # In the deltas, it seems some blocks move up by 3 rows.
        # Find all pixels of color 9 and move them up by 3.
        mask = (grid == 9)
        coords = np.argwhere(mask)
        for r, c in coords:
            new_grid[r, c] = 5
            if r - 3 >= 0:
                new_grid[r-3, c] = 9
        # Also handle the counter at r63 for Action 1.
        # We see r63c55 -> r63c54 etc.
        row63 = grid[63, :]
        one_pos = np.where(row63 == 1)[0]
        if len(one_pos) > 0:
            old_pos = one_pos[0]
            new_grid[63, old_pos] = 15
            # The observedACTION1 delta shows r63c55 -> r63c54.
            # This is also a shift left by 1 on row 63.
            if old_pos - 1 >= 0:
                new_grid[63, old_pos - 1] = 1

    elif action == 5:
        # ACTION5 changes r24c48 to 9 and r27c21 to 0 (or similar).
        # It seems like it's a "confirm" or "interact" action.
        # Based on deltas: r24c48 becomes 9, r27c21 becomes 0, r63c56 becomes 1.
        # We implement the specific change seen in the data.
        new_grid[24, 48] = 9
        new_grid[27, 21] = 0
        # Update counter at r63.
        row63 = grid[63, :]
        one_pos = np.where(row63 == 1)[0]
        if len(one_pos) > 0:
            old_pos = one_pos[0]
            new_grid[63, old_pos] = 15
            if old_pos - 1 >= 0:
                new_grid[63, old_pos - 1] = 1
        else:
            # If no '1' is found, just set a default position based on observed delta.
            new_grid[63, 56] = 1

    return new_grid

def is_level_complete(grid):
    # Level complete usually happens when an object reaches a target or a pattern is formed.
    # In this game, it seems to be related to color 0 (the hole/target).
    # We check if any pixel of color 0 has been filled by another color or if a specific state is reached.
    # Check if the bottom row counter has reached a certain point.
    # Or simply check for the presence of color 0 in the same place as a moving part.
    # Based on common ARC patterns, we look for the "win" condition.
    # Since no win grid was provided, we assume completion when the counter at r63 reaches c0.
    row63 = grid[63, :]
    one_pos = np.where(row63 == 1)[0]
    if len(one_pos) > 0:
        return one_pos[0] == 0
    return False