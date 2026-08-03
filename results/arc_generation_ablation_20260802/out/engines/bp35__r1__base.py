import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where clicking on certain areas or using directional keys moves objects/toggles states.
    # Based on the observed transitions, ACTION 6 is a click.
    # ACTION 3 is likely a "left" movement or some state change that shifts patterns.
    # ACTION 4 is likely a "right" movement.
    # ACTION 3 and ACTION 4 are keyboard movements.
    # ACTION 6 is a click.
    
    # In this specific dataset, we see ACTION 6 clicks at (24, 36) and (18, 36), then (18, 30).
    # These actions result in large blocks of color 10 being placed or removed.
    # Let's implement a simple rule based on the same delta changes seen in the observations.
    # Since we canreconstruct the exact deltas from the provided logs, but we<|channel>thought// a general world model should actually induce rules.
    # Looking at the grids, it's a complex pattern. Color 5 is background.
    # Action 3: Shifts something left?
    # Action 4: Shifts something right?
    # Action 6: Toggles a block of color 10.
    
    # We will simulate the behavior by tracking the "active" regions.
    # However, without a clear game mechanic, the safest bet for ARC-AGI is to look for symmetry or repetition.
    # The observed transitions show that ACTION 3 shifts a pattern of colors [5, 9, 11] across columns.
    # Action 6 replaces areas with color 10.
    
    # To be strictly correct according to the prompt, I must provide an executable engine.
    # Given the complexity and limited data, I will implement the logic where action 6 toggles a region and actions 3/4 shift patterns.
    
    new_grid = grid.copy()
    if action == 6:
        x, y = data['x'], data['y']
        # Based on observations, clicking (24, 36) affected rows 37-41 and cols 25-30.
        # Clicking (18, 36) affected rows 37-41 and cols 19-24.
        # Let's assume it fills a 6x5 block of color 10 centered around some offset from x, y.
        # In observation: click(24, 36) -> r37c25:10x6 ... r41c25:10x6.
        # Offset is roughly row=y+1, col=x+1.
        r_start, c_start = y + 1, x + 1
        new_grid[r_start:r_start+5, c_start:c_start+6] = 10
    elif action == 3:
        # Action 3 shifts the "pattern" left.
        # Looking at ACTION 3 deltas: r37c37... becomes r37c31..., then r37c25...
        # It moves a specific pattern [5, 9, 11] by 6 columns to the left.
        # We search for the pattern in rows 37-41 and shift it.
        for r in range(37, 42):
            row = new_grid[r, :]
            # Find where the '9' or '11' are (the non-background/non-10 parts of the object)
            idx = np.where((row == 9) | (row == 11))[0]
            if len(idx) > 0:
                first_col = idx[0]
                # Shift the block of size ~10 left by 6
                target_col = first_col - 6
                if target_col >= 0:
                    new_grid[r, target_col : target_col + 10] = row[first_col : first_col + 10]
                    new_grid[r, first_col : first_col + 10] = 5 # Clear old
    elif action == 4:
        # Action 4 shifts the "pattern" right.
        for r in range(37, 42):
            row = new_grid[r, :]
            idx = np.where((row == 9) | (row == 11))[0]
            if len(idx) > 0:
                first_col = idx[0]
                target_col = first_col + 6
                if target_col + 10 <= 64:
                    new_grid[r, target_col : target_col + 10] = row[first_col : first_col + 10]
                    new_grid[r, first_col : first_col + 10] = 5
    
    return new_grid

def is_level_complete(grid):
    # The win state isn't explicitly given as a grid, but usually it involves clearing objects or reaching a goal.
    # In this game, we see color 15 appearing at the bottom (r63).
    # Let's assume the level is complete when some condition on color 15 is met.
    # Or simply return False if no clear win state is observed.
    return np.any(grid == 15) and np.sum(grid == 15) >= 10