import numpy as np

def engine(grid, action, data=None):
    """
    Predicts the next state of the grid based on the given action.
    Action 4 (Right) moves a 'player' object (a vertical span of color 0/10/8/14)
    across the same pattern of obstacles/paths.
    Based on observed transitions, Action 4 shifts the entire active region right by 3 columns.
    The 'active region' consists of the rows 14-31 where values are changing.
    Specifically, the player is represented by the gaps (color 0) in the walls (color 10).
    When moving right, the gap at (row, col) becomes wall (10), and the new gap appears at (row, col+3).
    Additionally, a single pixel at r0c[col] changes from 10 to 0 or vice versa.
    """
    out = grid.copy()
    if action == 4:
        # The movement shift is 3 units.
        shift = 3
        
        # We need to find the current position of the "gap" (the player)
        # In row 14, the gap starts at column 11.
        # Let's determine the current x-position based on the pixels at r0.
        # Initial state: r0c16=4, but it seems the marker is actually shifting.
        # Looking at ACTION4 deltas:
        # Transition 1: r0c16:0x1, r14c11:10x3...
        # Transition 2: r14c14:10x3, r14c29:0x3...
        # Wait, looking closer at the same pattern:
        # Action 4 shifts the 'empty space' (color 0) right by 3 columns.
        # Row 14-16: Gap of 3 cells wide.
        # Row 17-19: Two gaps of 3 cells wide separated by wall.
        # Row 20-22: One gap of 3 cells wide and one wall of 3 cells wide.
        # Row 23-28: Gaps are shifted relative to each other.
        # Row 29-31: Gaps might be color 8 or 14.
        
        # To implement this simply, we identify all cells that change in a row.
        # The current "player" position can be determined by the same logic as 
        # the movement shift.
        
        # Find the column index where the first gap starts in row 14.
        # find the first occurrence of 0 in row 14 from col 11 onwards.
        try:
            gap_col = np.where(grid[14, 11:] == 0)[0][0] + 11
        except IndexError:
            # Fallback if no gap found (e.g., it's already at the end)
            return out

        # For every row from 14 to 31, we move the 'empty space' pattern.
        for r in range(14, 32):
            row_data = grid[r].copy()
            # We need to move the gaps.
            # This is essentially shifting the values in columns [gap_col : gap_col+6] 
            # and [gap_col+15 : gap_col+21] etc.
            # But looking at the evidence, only specific spans are 3 wide.
            # The a-priori layout is that the player is a complex shape made of gaps.
            # The "player" is everything that isn't color 10 (wall).
    
        # Instead of a more general rule, let's use the observed deltas directly.
        # Action 4 shifts the entire structure right by 3 units.
        # Let's find where the current x-position marker is on row 0.
        # Row 0 has a block of color 4. Color 4 is the background/path.
        # Row 0: 10x16, 4x32, 10x16. Gap is from col 16 to 47.
        # In ACTION4 transitions, r0c16 becomes 0, then r0c17 becomes 0, etc.
        # This is a bit strange. Let's look at the same pixel changing.
        # Transition 1: r0c16:0x1, r14c11:10x3...
        # Transition 2: r0c17:0x1? No, only r14 changes.
        # Wait, looking at the evidence again:
        # Trans 1: r0c16:0x1
        # Trans 2: (no r0 change)
        # Trans 3: r0c17:0x1
        # Trans 4: (no r0 change)
        # Trans 5: r0c18:0x1
        # Trans 6: (no r0 change)
        # Trans 7: r0c19:0x1
        # It seems every second Action 4 moves the marker on row 0 by 1 pixel.
        
        # Find current marker position on row 0.
        marker_col = np.where(grid[0] == 0)[0]
        if len(marker_col) > 0:
            curr_marker = marker_col[0]
        else:
            curr_marker = 16 # Initial state
            
        # The player's x-position is determined by the gap in row 14.
        # find first 0 in row 14 from col 11 onwards.
        gap_col = np.where(grid[14, 11:] == 0)[0][0] + 11 if any(grid[14, 11:] == 0) else 11
        
        # For each row that changes, we shift the "player" shape.
        # We identify the "player" as cells that are not color 10.
        #<|channel>thought
        # Let's just implement a simple shift of the gaps (color 0/8/14).
        # In rows 14-31, the 'player' consists of columns where grid[r, c] != 10.
        # To move right, we make current non-wall cells wall (10), and new non-wall cells (the ones at c+3) non-wall.
        
        for r in range(14, 32):
            # Find all indices where it's NOT a wall (10)
            non_walls = np.where(grid[r, 11:48] == 10)[0] # This is wrong.
            # Correct logic: find everything that ISN'T 10.
            players = np.where(grid[r, 11:48] != 10)[0] + 11
            
            # The player moves from col c to c+3.
            # Current players become walls.
            out[r, players] = 10
            # New positions become what they were.
            # Since the "path" is consistent, we just need to know what value was there.
            # But looking at deltas, the values are often 0 or 8 or 14.
            # Let's assume the movement is a simple shift of the 'gap' pattern.
            # We can use the grid state from 3 columns back if we had it, but we don't.
            # However, the evidence shows the new gaps are color 0 in most rows.
            # In row 29-31, some are 8 or 14.
            # Let's look at the delta for r29c14: 10x3, 8x3, 10x3, 8x3.
            # This means current gap (col 14-16) becomes wall (10), then next block (17-19) becomes 8...
            # Actually, it looks like the entire content of the row [11:48] is shifting right by 3.
            
            # Shift the slice [11:48] right by 3.
            slice_vals = grid[r, 11:48].copy()
            shifted = np.roll(slice_vals, 3)
            # But roll wraps around. The deltas show specific cells changing.
            # Let's just apply a simple shift and fill the start with walls.
            out[r, 11:48] = shifted
            out[r, 11:11+3] = 10

        # Update marker on row 0 every second move? Or maybe based on x position.
        # Transition 1: r0c16 -> 0. Gap was at col 11. New gap at 14.
        # Transition 3: r0c17 -> 0. Gap was at 14. New gap at 17.
        # It seems for every 3 units of movement in rows 14-31, the marker moves 1 unit in row 0.
        if (gap_col // 3) % 2 == 0: # This is a guess.
             pass
        
        # Let's refine the marker logic: it changes when gap_col increases by 3.
        # Initial gap_col = 11. Marker = none.
        # Move 1: gap_col = 14. Marker = r0c16=0.
        # Move 2: gap_col = 17. Marker = no change? No, Trans 2 has no r0 change.
        # Move 3: gap_col = 20. Marker = r0c17=0.
        # Move 4: gap_col = 23. Marker = no change.
        # Move 5: gap_col = 26. Marker = r0c18=0.
        # Move 6: gap_col = 29. Marker = no change.
        # Move 7: gap_col = 32. Marker = r0c19=0.
        # Rule: if move_count is odd, increment marker_col and set to 0.
        # But we don't have move_count. We can use (gap_col - 11) // 3.
        move_num = (gap_col - 11) // 3 + 1
        if move_num % 2 != 0:
            marker_pos = 16 + (move_num // 2)
            out[0, marker_pos] = 0

    return out

def is_level_complete(grid):
    # The level is complete when the player reaches a certain position.
    # In this game, it's likely when the gaps reach the right side of the board.
    # Let's check if any gap in row 14 has reached col 40+.
    try:
        gap_col = np.where(grid[14, 11:] == 0)[0][0] + 11
        return gap_col >= 40
    except IndexError:
        return False