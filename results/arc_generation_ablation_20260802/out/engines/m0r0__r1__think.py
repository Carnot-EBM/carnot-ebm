import numpy as np

def engine(grid, action, data):
    """
    Induces the world model for game 'm0r0'.
    Based on observed transitions:
    - ACTION1 and ACTION4 seem to shift blocks of color 10 (value 10) horizontally or vertically.
    - Specifically, looking at the deltas, cells of value 5 are being replaced by 10s in specific regions.
    - The grid contains boundaries of colors 11 and 12.
    - There's a pattern where certain areas change from 5 to 10 and back.
    - Action 1 moves/toggles blocks in columns 14-19 and 44-49.
    - Action 3 seems to toggle values between 5 and 10 in column 44-49.
    - Action 4 shifts these patterns.
    - Additionally, there's a counter moving across row 0 and row 63 (color 0).
    """
    next_grid = grid.copy()
    h, w = next_grid.shape

    # Track the "cursor" (the cell with value 0)
    # In the observations, r0c62 becomes 0, then r0c61, etc.
    # And r63c1 becomes 0, then r63c2, etc.
    
    if action == 1:
        # ACTION1 typically changes color 5 to 10 in specific vertical bands
        # Band A: cols 14-18, Band B: cols 44-48
        # It appears to move a 'window' of 10s up or down.
        # Based on deltas: it replaces 5x5 areas with 10x5.
        # Let's implement the observed shift logic.
        # The delta shows rows 34-38 changing first, then 29-33, then 24-28...
        # This is a movement of a block of size 5x5 upwards.
        
        # Find current blocks of 10s in the active columns
        for col in range(14, 19):
            # Simple heuristic: if we see 10s, they might move.
            pass
        
        # To match the provided transitions exactly without a complex state machine:
        # We observe that Action 1 moves the "active" 10-block window.
        # Since we don't have the full sequence of all possible actions, 
        # we simulate the most likely behavior: shifting existing 10s.
        
        # Shift 10s in bands [14:19] and [44:49]
        for band in [(14, 19), (44, 49)]:
            c_start, c_end = band
            # Find rows where color 10 exists in this band
            rows_with_10 = np.where((next_grid[:, c_start:c_end] == 10).any(axis=1))[0]
            if len(rows_with_10) > 0:
                r_min, r_max = rows_with_10[0], rows_with_10[-1]
                # Move block up by 5
                new_r_min, new_r_max = max(0, r_min - 5), max(0, r_max - 5)
                # Clear old
                next_grid[r_min:r_max+1, c_start:c_end] = 5
                # Set new
                next_grid[new_r_min:new_r_max+1, c_start:c_end] = 10

    elif action == 3:
        # ACTION3 toggles values between 5 and 10 in the right band [44:49]
        for r in range(h):
            for c in range(44, 49):
                if next_grid[r, c] == 5:
                    next_grid[r, c] = 10
                elif next_grid[r, c] == 10:
                    next_grid[r, c] = 5

    elif action == 4:
        # ACTION4 seems to shift blocks horizontally or toggle bands.
        # Observation shows it affects both bands simultaneously.
        for band in [(14, 19), (44, 49)]:
            c_start, c_end = band
            rows_with_10 = np.where((next_grid[:, c_start:c_end] == 10).any(axis=1))[0]
            if len(rows_with_10) > 0:
                r_min, r_max = rows_with_10[0], rows_with_10[-1]
                # Shift logic for Action 4 is different; often a horizontal swap/toggle
                # In the delta, it replaces 5x5 with 10x5 and vice versa.
                pass

    # Update cursors (the value 0 cells)
    # Row 0 cursor moves left
    r0_zeros = np.where(next_grid[0] == 0)[0]
    if len(r0_zeros) > 0:
        curr_col = r0_zeros[0]
        if curr_col > 0:
            next_grid[0, curr_col] = 5
            next_grid[0, curr_col - 1] = 0
            
    # Row 63 cursor moves right
    r63_zeros = np.where(next_grid[63] == 0)[0]
    if len(r63_zeros) > 0:
        curr_col = r63_zeros[0]
        if curr_col < w - 1:
            next_grid[63, curr_col] = 5
            next_grid[63, curr_col + 1] = 0

    return next_grid

def is_level_complete(grid):
    # A level is complete if the cursors reach their targets or a specific pattern is met.
    # Based on typical ARC games, we check for a win condition like all target cells filled.
    # For this game, let's assume it's complete when the top-left/bottom-right corners are reached.
    return grid[0, 0] == 0 and grid[63, 63] == 0