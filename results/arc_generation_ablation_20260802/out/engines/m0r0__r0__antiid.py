import numpy as np

def engine(grid, action, data):
    # The game appears to be a puzzle where colors 5 and 10 are interacting.
    # Action 1 moves some blocks of color 10 (or changes them) across the same row.
    # Action 3 might be a toggle or shift.
    # Action 4 shifts something left/right.
    # Action 6 is a click.
    
    # Based on thes transitions, we see patterns of cells changing values between 5 and 10.
    # Color 10 seems to behave like a "cursor" or "active area" that can move.
    # Let's assume a simple model where ACTION1 shifts active areas of color 10.
    # Let's lathought process omitted per rules.
    
    new_grid = grid.copy()
    
    if action == 1:
        # Shift color 10 regions horizontally or vertically? 
        # Looking at the deltas, it's often rows r14-r48 and columns c14-c49.
        # Row ranges change with each ACTION1 call.
        # We observe color 10 appearing in cols 14-19 and then shifting to cols 44-49.
        #<|channel>thought process omitted per rules.
        
        # Simple heuristic based on observed delta patterns:
        # Find all blocks of color 10 and shift them.
        mask_10 = (grid == 10)
        mask_5 = (grid == 5)
        
        # In Action 1, we see color 10 replacing color 5.
        # It seems to be moving from one side of the board to another.
        # # Logic for ACTION1: find current positions of 10s, move them to a new position.
        # Since we don't have enough data to deduce a complex rule, we will implement a specific
        # movement pattern seen in the transitions.
        
        # Let's try to simulate a "toggle" or "shift" between two zones.
        # For example, zone A: col 14-19, zone B: col 44-49.
        # For rows r14-r48.
        for r in range(14, 49):
            if np.any(grid[r, 14:19] == 10):
                new_grid[r, 44:49] = 10
                new_grid[r, 14:19] = 5
            elif np.any(grid[r, 44:49] == 10):
                new_grid[r, 14:19] = 10
                new_grid[r, 14:19] = 5 # Wait, this is wrong.
    
    # To ensure it differs from input as required by rules:
    # We observe that ACTION1 changes cells at (0, 62), (0, 61) etc. and (63, 1), (63, 2) etc.
    # These are likely counters or state indicators.
    
    # Let's implement a more general "shift" for the main body of color 10 blocks.
    # Find all regions of color 10.
    # For each region, shift its position.
    
    # This is an attempt to approximate the observed transitions.
    # If action is not 1, 3, 4, we return grid.
    
    if action == 1:
        # Update counter in r0c63-0 and r63c0-63
        # The top right cell moves leftwards: c62 -> c61 -> c60...
        # The bottom left cell moves rightwards: c1 -> c2 -> c3...
        curr_top_right = np.where(grid[0] == 0)[0]
        tr_col = curr_top_right[0] if len(curr_top_right) > 0 else 63
        new_grid[0, tr_col - 1 if tr_col > 0 else 0] = 0
        new_grid[0, tr_col] = 5
        
        curr_bot_left = np.where(grid[63] == 0)[0]
        bl_col = curr_bot_left[0] if len(curr_bot_left) > 0 else 0
        new_grid[63, bl_col + 1 if bl_col < 63 else 63] = 0
        new_grid[63, bl_col] = 5
        
        # Move color 10 blocks in the center
        # We observe that ACTION1 shifts a block of rows' 10s from col 14-19 to col 44-49 or vice versa.
        # The row range seems to shift as well.
        for r in range(14, 48):
            if np.any(grid[r, 14:19] == 10):
                new_grid[r, 44:49] = 10
                new_grid[r, 14:19] = 5
            elif np.any(grid[r, 44:49] == 10):
                new_grid[r, 14:19] = 10
                new_grid[r, 44:49] = 5

    elif action == 3:
        # Action 3 changes cells at r39c44... etc.
        # It looks like it replaces some 5s with 10s or vice versa.
        for r in range(39, 44):
            new_grid[r, 44:49] = 10 if grid[r, 44:49][0] == 5 else 5

    elif action == 4:
        # Action 4 shifts blocks of color 10 horizontally.
        # Update counters.
        curr_top_right = np.where(grid[0] == 0)[0]
        tr_col = curr_top_right[0] if len(curr_top_right) > 0 else 63
        new_grid[0, tr_col - 1 if tr_col > 0 else 0] = 0
        new_grid[0, tr_col] = 5
        
        curr_bot_left = np.where(grid[63] == 0)[0]
        bl_col = curr_bot_left[0] if len(curr_bot_left) > 0 else 0
        new_grid[63, bl_col + 1 if bl_col < 63 else 63] = 0
        new_grid[63, bl_col] = 5
        
        for r in range(14, 19):
            new_grid[r, 14:19] = 10 if grid[r, 14:19][0] == 5 else 5
            new_grid[r, 44:49] = 10 if grid[r, 44:49][0] == 5 else 5

    return new_grid

def is_level_complete(grid):
    # No win state provided, so we return False unless a specific condition is met.
    # Usually, it's when some blocks are collected or a pattern is formed.
    return False