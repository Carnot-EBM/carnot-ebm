import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action == 4: # ACTION4 seems to be a "right" movement of some object/pattern
        new_grid = grid.copy()
        # The observed transitions show a pattern shifting right by 3 columns.
        # Specifically, it's replacing blocks of color 0 and moving them.
        # We need to find the same logic that governs this shift.
        # Looking at the delta, it's not just a simple translation.
        # It's more than like a "cursor" or "player" moving across a gap.
        # Let's try to implement a shift of the specific colors involved in the 0-regions.
        # 
        # Based on the<|channel>thought process, let's assume there is an active region 
        # defined by the gaps (color 0) and the boundaries (color 10).
        # 
        # In the provided deltas, the cells being changed are rows 14-31.
        # Rows 0-13 and 32-63 are mostly unchanged except for r0c16...r0c19.
        # Row 0 has a block of color 4.
        # Color 4 is acting as a ceiling.
    
    # Since we only have ACTION4, and no other actions, 
    # and the observed transitions show a pattern shifting right by 3 columns,
    # let's generalize:
    # Find all regions of color 0 and shift them? No.
    # The most consistent thing is that a set of blocks is moving right.
    # Let's look at the delta again.
    # r14c11:10x3 means row 14, col 11 becomes color 10 for 3 cells.
    # r14c26:0x3 means row 14, col 26 becomes color 0 for 3 cells.
    # This looks like a "block" of size (H=18, W=3) shifted from x=26 to x=11? 
    # Wait, if it moves RIGHT, then the old position should become background and new position should be object.
    # If action 4 is 'Right', then something at x was moved to x+3.
    # Old pos x -> background; New pos x+3 -> object.
    # In the first transition: r14c11:10x3 (new object), r14c26:0x3 (new background).
    # That's a shift of 15 columns? No, let's re-read.
    # Transition 1: r14c11:10x3, r14c26:0x3. Shift = 26 - 11 = 15.
    # Transition 2: r14c14:10x3, r14c29:0x3. Shift = 29 - 14 = 15.
    # Transition 3: r14c17:10x3, r14c32:0x3. Shift = 32 - 17 = 15.
    # Each ACTION4 shifts the "object" by 3 pixels to the right.
    # The "object" is the set of cells that are NOT color 0 in the gap region.
    # Let's identify the "gap region": rows 14-31, cols 11-47 approx.
    # In this region, we move all non-zero values 3 units to the right and fill the vacated space with zero?
    # Or rather, it looks like a specific pattern is moving.
    # Let's try a simpler approach: shift everything in the "active area" (rows 14-31) 3 columns right.
    
    if action == 4:
        new_grid = grid.copy()
        # Active area based on observed deltas
        r_start, r_end = 14, 32
        c_start, c_end = 11, 48 # Approximate boundaries of the 'hole'
        
        # We need to find what exactly is moving.
        # It seems the blocks of color 10 and 8 are shifting.
        # Let's just shift the entire slice [r_start:r_end, c_start:c_end] by 3.
        slice_data = new_grid[r_start:r_end, c_start:c_end].copy()
        # Shift right by 3
        shifted = np.roll(slice_data, 3, axis=1)
        # Fill the first 3 columns with the background color (which is usually 10 or 0 in this game)
        # Looking at the delta: r14c11 becomes 10x3. So fill with 10.
        shifted[:, :3] = 10 
        # The cells that were shifted out (last 3 cols) should be filled with 0?
        # Delta says r14c26:0x3. This means the "gap" (color 0) is also moving.
        # Actually, it looks like a window of size 3 is sliding.
        
        # Let's try to find the 'cursor' position from row 0.
        # Transition 1: r0c16:0x1. Transition 2: r0c17:0x1...
        # Row 0 has color 4 block. The index of the 0 cell in row 0 is the cursor.
        cursor = np.where(grid[0] == 0)[0]
        if len(cursor) > 0:
            curr_x = cursor[0]
            new_grid[0, curr_x] = 4 # Restore old
            new_grid[0, curr_x + 1] = 0 # Move cursor right
            
            # Now move the pattern in rows 14-31.
            # The delta shows blocks of 3 columns are changing.
            # It seems we shift all non-zero values in that region by 3?
            # No, let's just use the observed behavior: 
            # for each row in [14, 31], if there's a block of 0s, it shifts.
            # This is getting complex. Let's simplify: 
            # ACTION 4 moves everything in rows 14-31 three units to the right.
            for r in range(r_start, r_end):
                row = new_grid[r, c_start:c_end].copy()
                shifted_row = np.roll(row, 3)
                shifted_row[:3] = 10 # Fill with boundary color
                new_grid[r, c_start:c_end] = shifted_row
        return new_grid

    return grid

def is_level_complete(grid):
    # Win state usually involves clearing something or reaching a position.
    # We don't have a win state example, but typically it's when the cursor reaches the end.
    cursor = np.where(grid[0] == 0)[0]
    if len(cursor) > 0 and cursor[0] >= 63:
        return True
    return False