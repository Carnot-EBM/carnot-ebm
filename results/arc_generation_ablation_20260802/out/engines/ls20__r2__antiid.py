import numpy as np

def engine(grid, action, data):
    # Grid size 64x64
    # Based on the observations, ACTION1 (Up), ACTION3 (Left), ACTION4 (Right)
    # seem to move some object/entity represented by color 3 (or related colors)
    # and potentially update a status bar at the bottom (rows 61-62).
    # The entity seems to be a set of pixels of color 3 (blue) or others.
    # Let's identify the 'player' position based on the rows 61-62 indicator.
    # The same cells (r61c14, r62c14) etc. change when moving.
    #
    # Looking at the transitions:
    # ACTION3 (Left): moves something from c24 to c19.
    # ACTION4 (Right): moves something from c19 to c24.
    # ACTION1 (Up): moves something from r45 to r40, then r40 to r35, etc.
    # ACTION1 also updates the cursor in rows 61-62.
    # ACTION1 (level 0->0): changed cells = r40c19:12x5... r61c16:3x1 r62c16:3x1
    # It looks like there is a "cursor" or "pointer" that tracks movement.
    #
    # We need to find where the current pointer/entity is located.
    # In the initial grid, we can actually see some patterns of color 3.
    #
    # Let's define the object as the block of colors starting around row 8 and ending row 16.
    # The 'active' part seems to be the same set of columns.
    # # Simplified logic:
    # # Find the coordinates of the pixels of color 3.
    # # lathought:
    # # Action 1: Up. Shift the active region up by 5 units.
    # # Action 3: Left. Shift the active region left by 5 units.
    # # Action 4: Right. Shift the active region right by 5 units.
    # # The status bar at the bottom (rows 61-62) also moves its cursor.
    # # ACTION1 (Up): cursor shifts right? (r61c16, r62c16 then r61c17, etc.)
    # # ACTION4 (Right): cursor shifts right.
    # # ACTION3 (Left): cursor shifts left.
    # #
    # # We need to find the current "cursor" position in rows 61-62.
    # # Find where color 3 is in rows 61-62.
    # # In initial grid: r61c14:3x1, r62c14:3x1.
    #<|channel>thought
    # Let's implement a movement rule based on the observed deltas.

    new_grid = grid.copy()
    
    # Identify the 'active block' - it seems to be a set of pixels that move.
    # Based on observations, they are often colors 3, 9, 12.
    # These blocks are usually 5 columns wide and several rows high.
    # The movements are in increments of 5.
    
    if action == 1: # Up
        # Shift some region up by 5.
        # Looking at the data: r40-r49 moves to r35-r44.
        # This looks like a vertical shift of a specific pattern.
        # We search for the block of color 3/9/12 and shift it.
        # For simplicity, we can find all cells with values not in [0, 4, 5] (the background)
        # and if they are part of the moving object, shift them.
        mask = (grid != 0) & (grid != 4) & (grid != 5)
        coords = np.argwhere(mask)
        if len(coords) > 0:
            # Find the bounding box or just shift everything that isn't static background.
            # However, there is also a status bar. Let's only shift things above row 60.
            for r, c in coords:
                if r < 60:
                    new_grid[r-5, c] = grid[r, c]
                    new_grid[r, c] = 4 # Reset to background
            # Update cursor in rows 61-62. It shifts right by 1 on ACTION1?
            # Initial: r61c14, r62c14 -> Action 1: r61c16, r62c16. Wait, +2?
            # Let's check: r61c14 -> r61c16 -> r61c17... it varies.
            # Actually, let's look at the delta again: r61c16:3x1, r62c16:3x1.
            # That means cell (61, 16) becomes color 3.
            cursor_cols = np.where((grid[61] == 3))[0]
            if len(cursor_cols) > 0:
                curr_c = cursor_cols[0]
                new_grid[61, curr_c] = 11 # Clear old
                new_grid[62, curr_c] = 11
                new_grid[61, curr_c + 1] = 3
                new_grid[62, curr_c + 1] = 3

    elif action == 3: # Left
        mask = (grid != 0) & (grid != 4) & (grid != 5)
        coords = np.argwhere(mask)
        for r, c in coords:
            if r < 60:
                new_grid[r, c-5] = grid[r, c]
                new_grid[r, c] = 4
        cursor_cols = np.where((grid[61] == 3))[0]
        if len(cursor_cols) > 0:
            curr_c = cursor_cols[0]
            new_grid[61, curr_c] = 11
            new_grid[62, curr_c] = 11
            new_grid[61, curr_c - 1] = 3
            new_grid[62, curr_c - 1] = 3

    elif action == 4: # Right
        mask = (grid != 0) & (grid != 4) & (grid != 5)
        coords = np.argwhere(mask)
        for r, c in coords:
            if r < 60:
                new_grid[r, c+5] = grid[r, c]
                new_grid[r, c] = 4
        cursor_cols = np.where((grid[61] == 3))[0]
        if len(cursor_cols) > 0:
            curr_c = cursor_cols[0]
            new_grid[61, curr_c] = 11
            new_grid[62, curr_c] = 11
            new_grid[61, curr_c + 1] = 3
            new_grid[62, curr_c + 1] = 3

    return new_grid

def is_level_complete(grid):
    # No win state provided, but typically it's when the object reaches a target.
    # Let's assume it's complete if the object moves to a specific area.
    return False