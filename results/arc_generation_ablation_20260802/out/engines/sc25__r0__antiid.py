import numpy as np

def engine(grid, action, data):
    # The game state consists of a 64x64 grid.
    # Action 3 (Left) moves a set of blocks (colors 9, 10, 2) moving leftwards.
    # Based on the transitions, ACTION3 shifts some elements horizontally.
    # Specifically, it seems to move a pattern [9, 10, 2, 2] across columns.
    # In the same time, it changes cells at column 62 and 63 (color 14 -> 0).
    # Looking at the transition deltas, we see r19c37->r19c35->r19c33...
    # It's shifting by 2 units each time.
    
    new_grid = grid.copy()
    
    if action == 3:
        # Identify the "player" or "active object" which is the sequence [9, 10, 2, 2]
        # We look for the first occurrence of color 9 in rows 19 and 20.
        #<|channel>thought
        # Let's find where the block starts.
        for r in [19, 20]:
            # Find current position of color 9
            cols = np.where(grid[r] == 9)[0]
            if len(cols) > 0:
                curr_col = cols[0]
                # Move it left by 2
                new_col = curr_col - 2
                if new_col >= 0:
                    # The pattern observed in delta is 9x1, 10x1, 2x2 (total width 4)
                    # Shift the whole 4-cell block
                    # Clear old position
                    new_grid[r, curr_col : curr_col + 4] = 5 # Background color
                    # Place new position
                    new_grid[r, new_col : new_col + 4] = [9, 10, 2, 2]
    
    # Additionally, some transitions show column 62/63 changing from 14 to 0.
    # This looks like a "timer" or "progress bar" being consumed.
    # We need to simulate this consumption based on the action.
    # Let's find rows where col 62 and 63 are 14.
    # In ACTION3, we see r6c62:0x2, r7c62:0x2 etc.
    # These happen in pairs of rows.
    # For each ACTION3 call, it seems to consume two rows of the border at x=62.
    # To make it deterministic and consistent with observed deltas:
    # We assume there is a state variable not explicitly stored in grid but implied by current layout.
    # Find first row that still has 14s at (r, 62) and (r, 63).
    # Find all rows that have 14s at (r, 62) and (r, 63).
    # Find rows that already became 0.
    # The same pattern repeats for every ACTION3.
    
    # Since we don't have a global state, we look for the first pair of rows starting from top.
    rows_with_14 = np.where(grid[:, 62] == 14)[0]
    if len(rows_with_14) > 0:
        # Consume the first two available rows of color 14
        for i in range(min(2, len(rows_with_14))):
            row = rows_with_14[i]
            new_grid[row, 62:64] = 0
            
    return new_grid

def is_level_complete(grid):
    # Level complete when the border of 14s is gone or player reaches target.
    # No more 14s on the right edge.
    return not np.any(grid[:, 62:64] == 14)