import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action 3 is a movement action that moves an object (color 2) and changes some cells on the right edge.
    # Based on observed transitions, ACTION3 shifts a specific object (the color 2 rectangle/block) leftwards.
    # It also seems to actually change colors of cells at column 62-63 (right edge) to 0.
    # The same pattern repeats in rows 19 and 20.
    
    new_grid = grid.copy()
    if action == 3:
        # Find the "player" or moving object (color 2) in rows 19-20.
        # In the initial state, it's a block of color 2 starting around x=17 to 42.
        # Each ACTION3 call shifts the block of color 2 from rows 19-20 left by 2 columns.
        # Shift the block of color 2 in rows 19-20 specifically.
        # For r19 and r20, we find all indices where value is 2.
        #<|channel>thought
        for r in [19, 20]:
            row_vals = new_grid[r]
            mask = (row_vals == 2)
            if not np.any(mask):
                continue
            
            # Find the leftmost and rightmost bounds of the color 2 segment in these rows.
            # We only care about the shift for the specific block that was observed.
            # Find first and last index of color 2 in row r.
            indices = np.where(row_vals == 2)[0]
            if len(indices) > 0:
                first = indices[0]
                last = indices[-1]
                # Move this segment left by 2 units.
                # Fill original positions with background color (5).
                new_grid[r, first:last+1] = 5
                # Place shifted segment.
                # Ensure we don't go out of bounds.
                start = max(0, first - 2)
                end = max(0, last - 2)
                new_grid[r, start:end+1] = 2
                
        # Additionally, ACTION3 changes cells at column 62-63 to 0.
        # The transitions show a pattern: r6c62:0x2, r7c62:0x2, etc.
        # Each call to ACTION3 seems to clear two rows on the right edge.
        # We need to track which rows have been cleared.
        # Since we are not given state for "current row", we can count how many times action 3 has been called.
        # However, the engine must be pure. We can infer the current progress from the grid itself.
        # Find the same block in rows 19-20 and its position relative to initial.
        # # Initial x=17 (approx)
        # In observed transitions, the object moves left by 2 columns each time.
        # Let's find the current leftmost index of color 2 in row 19.
        indices = np.where(new_grid[19] == 2)[0]
        if len(indices) > 0:
            curr_left = indices[0]
            # Initial was around 17? No, looking at INITIAL GRID:
            # r19:5x12, 9x1, 10x4, 2x22... -> col 12+1+4 = 17.
            initial_left = 17
            dist = (initial_left - curr_left) // 2
            # The right edge clearing happens in pairs of rows starting from r6.
            # Rows cleared so far: (r6, r7), then (r8, r9), etc.
            # Each ACTION3 call clears the next pair.
            # We can determine which pair should be cleared now based on distance moved.
            # a bit complex for this game. Let's just apply the delta provided in observations.
            # If we are moving the block, clear the corresponding rows.
            # For each action 3, shift block left by 2 and clear two rows on the right.
            # To make it deterministic and pure, let's use the current position to decide.
            # This is tricky because the grid doesn't store "action count".
            # But wait, the observed transitions show that AFTER shifting, the cells change.
            # Let's simply find the first row at column 62 that is NOT 0 and set it and the one below it to 0.
            for r in range(6, 64):
                if new_grid[r, 62] != 0:
                    new_grid[r, 62:64] = 0
                    new_grid[r+1, 62:64] = 0
                    break
    return new_grid

def is_level_complete(grid):
    # Level complete when all color 14 (right edge) are replaced by 0?
    # Or some other condition. Usually, clearing a target area.
    # In this game, ACTION3 clears the right edge.
    return np.all(grid[6:64, 62:64] == 0)