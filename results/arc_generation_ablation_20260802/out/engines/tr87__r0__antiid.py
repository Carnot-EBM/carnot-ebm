import numpy as np

def engine(grid, action, data):
    # The game seems to involve moving blocks/colors on a 64x64 grid.
    # Based on the transitions, ACTION4 moves something at rows 48-60 and columns 15-40.
    # ACTION2 and ACTION1 move colors (5, 7) in the region r52-r56, c22-c33.
    # ACTION1 shifts things rightward or modifies them locally.
    # ACTION2 shifts things leftward or modifies them locally.
    # ACTION//C6 is click.
    
    new_grid = grid.copy()
    
    if action == 4:
        # Observed ACTION4 changes occur at specific column offsets.
        # It looks like it's shifting a pattern of color 0 and 3 across the same set of rows.
        # Find current position of color 0 in row 48.
        # We need to simulate the shift based on the observed deltas.
        # For example, first ACTION4 shifted from c15 to c22.
        #<|channel>thought
        # Let's try to find where the "active" block is.
        col_start = -1
        for c in range(64):
            if grid[48, c] == 0:
                col_start = c
                # break # Only one block?
        
        if col_start != -1:
            # Shift the block (columns col_start to col_start+4) by +7 columns.
            shift = 7
            # The delta shows r48c15:3x5, r48c22:0x5. This means old 0s become 3s, new 0s are placed.
            # Rows affected: 48, 49, 59, 60.
            # Row 49/59 have specific patterns.
            rows_to_move = [48, 49, 59, 60]
            
            # To implement this simply, we identify the current 'window' of color 0 and shift it.
            # We also need to handle row 49 and 59 which only change at two points.
            
            # Let's find all indices where grid[48, :] == 0.
            zeros = np.where(grid[48, :] == 0)[0]
            if len(zeros) > 0:
                curr_col = zeros[0]
                new_col = curr_col + 7
                if new_col < 64 - 5:
                    # Update rows 48 and 60 (full block move)
                    new_grid[48, curr_col : curr_col+5] = 3
                    new_grid[48, new_col : new_col+5] = 0
                    new_grid[60, curr_col : curr_col+5] = 3
                    new_grid[60, new_col : new_col+5] = 0
                    
                    # Row 49 and 59 are more complex. They have a gap.
                    # The delta shows r49c15:3x1, r49c19:3x1, r49c22:0x1, r49c26:0x1.
                    # This means indices [curr_col] and [curr_col+4] become 3, and [new_col] and [new_col+4] become 0.
                    new_grid[49, curr_col] = 3
                    new_grid[49, curr_col + 4] = 3
                    new_grid[49, new_col] = 0
                    new_grid[49, new_col + 4] = 0
                    new_grid[59, curr_col] = 3
                    new_grid[59, curr_col + 4] = 3
                    new_grid[59, new_col] = 0
                    new_grid[59, new_col + 4] = 0
            
    elif action == 2:
        # ACTION2 seems to move colors in the region r52-r56.
        # It looks like it's shifting a pattern of color 5/7.
        # Let's try to shift things leftward or modify them locally.
        # In the observed transitions, ACTION2 changes cells in c22-c26.
        # Find current position of something specific.
        # We will just apply a simple modification based on the delta if we can find a trigger.
        # # The delta shows r52c22:5x1, etc.
        # This is too complex for a general rule. Let's use a simpler "shift" logic.
        # Shift everything in rows 52-56 by -1 column?
        for r in range(52, 57):
            row = grid[r, :]
            new_grid[r, :] = np.roll(row, -1)
            # Keep boundaries fixed (color 3).
            new_grid[r, 0:14] = grid[r, 0:14]
            new_grid[r, 49:] = grid[r, 49:]

    elif action == 1:
        # ACTION1 seems to move colors in region r52-r56, shifting right.
        for r in range(52, 57):
            row = grid[r, :]
            new_grid[r, :] = np.roll(row, 1)
            # Keep boundaries fixed.
            new_grid[r, 0:14] = grid[r, 0:14]
            new_grid[r, 49:] = grid[r, 49:]

    # The observed transitions also show the cell r63c62, r63c61... changing color 4.
    # This looks like a counter or progress bar moving leftward.
    if action in [1, 2, 4]:
        # Find current position of color 4 in row 63.
        pos = np.where(grid[63, :] == 4)[0]
        if len(pos) > 0:
            curr_p = pos[0]
            if curr_p > 0:
                new_grid[63, curr_p] = 0 # Or whatever background is (usually 0 or 2).
                # Based on INITIAL GRID, r63 is mostly 1x63 then 4x1. So bg is 1? No, it's 1x63.
                # Let's use the value from the cell to its left.
                new_grid[63, curr_p - 1] = 4
                new_grid[63, curr_p] = 1 

    return new_grid

def is_level_complete(grid):
    # Win state not provided, but usually involves reaching a target or clearing blocks.
    # We can check if the "progress bar" at r63 has reached column 0.
    return np.any(grid[63, 0] == 4)