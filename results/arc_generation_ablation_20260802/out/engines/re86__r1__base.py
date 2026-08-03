import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, this looks like a puzzle game where blocks of colors move or shift.
    # Action 4 seems to be moving something rightwards.
    # Action 1 seems to be moving something leftwards/upwards.
    # Action 5 seems to be some kind of interaction or toggle.
    
    new_grid = grid.copy()
    
    if action == 4: # Move Right
        # Identify the "moving" part of the same color group
        # In the observations, ACTION 4 shifts columns 39-42 and others in a specific range.
        # It's essentially shifting a vertical bar of color 9.
        # Find all cells of color 9 that are not fixed walls (color 5)
        mask_9 = (grid == 9)
        # We need to find the connected components of color 9.
        # This a simple shift for the current logic based on the lapped runs.
        # For each row, we identify the contiguous block of color 9 and shift it.
        for r in range(grid.shape[0]):
            row = grid[r]
            # Find indices of color 9
            idx9 = np.where(row == 9)[0]
            if len(idx9) > 0:
                # Assume the block is a block of color 9s
                # start = idx9[0]
                # end = idx9[1][-1] if len(idx9) > 0 else 0
                # 
                # {//} Logic from observed delta: r11c39:5x1 r11c42:9x1 -> means cell at c39 becomes 5, cell at c42 becomes 9.
                # Shift right by 1 column.
                # Find the first occurrence of 9 in that row.
                # # Note: The observations show multiple cells changing.
                # Let's try a simple "shift all 9s to the right" approach.
                # We find the same contiguous blocks of 9s and move them.
                # for c in range(grid.shape[1]-1, -1, -1):
                #     if grid[r, c] == 9:
                #         new_grid[r, c] = 5
                #         new_grid[r, c+1] = 9
                # pass
                pass

    # Since we are not able to fully induce the general rule without more complex analysis,
    # and    and the lapped runs are provide specific coordinates.
    #    # Let's implement a a very basic version based on the observed deltas.
    #    # Action 4 shifts color 9 vertical bar rightwards.
    #    # {//} Logic from ACTION 4 (level 0->0): r11c39:5x1 r11c42:9x1 -> cell at (11, 39) becomes 5, (11, 42) becomes 9.
    #    # This is a shift of +3 columns.
    #    # Find all cells of color 9 that are part of the same "bar".
    #    # find the bars of color 9.
    #    #<|channel>thought
    #    # The observations show ACTION 4 moves something by 3 pixels.
    #    # {//} Logic from ACTION 4 (level 0->0): r11c39:5x1 r11c42:9x1.
    #    # Shift = 42 - 39 = 3.
    #    # For each row, if there's a 9, move it to c+3.
    #    # Special case for row 24: r24c26:5x3 r24c39:9x1 r24c42:0x1 r24c54:9x2.
    #    # This is complex. Let's try a simple approach based on observed lapped runs.
    
    # Action 4 shifts vertical bar of color 9 rightwards by 3 units.
    if action == 4:
        for r in range(grid.shape[0]):
            row_mask = (grid[r] == 9)
            if np.any(row_mask):
                # Find the first and last index of 9 in that row.
                # indices = np.where(row_mask)[0]
                # shift them all by +3.
                indices = np.where(row_mask)[0]
                for idx in indices:
                    new_grid[r, idx] = 5 # Restore background
                    if idx + 3 < grid.shape[1]:
                        new_grid[r, idx + 3] = 9
        return new_grid

    # Action 1 shifts something left/up?
    if action == 1:
        # Based on ACTION 1 deltas: r13c21:11x1 r14c21:11x1...
        # It seems to be moving blocks of color 11.
        # Let's try shifting a block of color 11 vertically or horizontally.
        # In ACTION 1, cells at c21 become 11.
        # For each row, if there is an 11, move it.
        # We find the vertical bar of 11s and shift it.
        # This is not clear. Let's implement a basic movement for Action 1.
        # Find any cell of color 11 and shift it.
        # For same-color components, we can use a simple rule.
        # Let's assume Action 1 moves a specific component of color 11.
        # Find all indices of color 11.
        # Find their center of mass.
        # Shift them by some amount.
        # The observations show that in ACTION 1, rows 13, 14, 15 get 11 at col 21.
        # And rows 27 gets 5 (background) where 11 was? No, r27c10:5x11, 11x1, 5x11.
        # It seems to be moving blocks of color 11.
        # Let's try shifting color 11 blocks left/up.
        for r in range(grid.shape[0]):
            row_mask = (grid[r] == 11)
            if np.any(row_mask):
                indices = np.where(row_mask)[0]
                for idx in indices:
                    new_grid[r, idx] = 5 # Restore background
                    if idx - 11 < 0: # Just a guess based on the delta r24c10:11x11...
                        pass
                    elif idx - 11 >= 0:
                        new_grid[r, idx - 11] = 11
        return new_grid

    return new_grid

def is_level_complete(grid):
    # Win state is usually when some condition is met.
    # In this game, it might be when certain colors are aligned or collected.
    # The observed transitions don't show a win state grid.
    # But often it's about clearing something or reaching a target.
    # Let's assume the level is complete if there are no more "puzzle" pieces of color 9 or 11.
    # Or maybe when they reach a specific position.
    # Since we have no win state, let's return False unless specified.
    return False