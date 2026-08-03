import numpy as np

def engine(grid, action, data):
    """
    The game 'lp85' appears to be a puzzle where ACTION 0 (or others) triggers 
    a cyclic shift of colors in specific regions of the grid.
    Based on the observed transitions, there are several blocks of cells that 
    change their values simultaneously. These changes look like permutations 
    of a set of colors {1, 2, 9, 10, 11, 15}.
    
    Specifically, looking at rows 19-22 and other designated areas, 
    the columns [12, 18, 24, 30, 36, 42, 48] (each width 4) undergo shifts.
    Also, column 0 undergoes a change every few actions.
    """
    new_grid = grid.copy()
    
    # The provided observations show ACTION 0 causing these shifts.
    if action == 0:
        # Column 0 logic: it seems to cycle through color 5 in chunks of 5 rows.
        # Transition 1: r0-r4 -> 5
        # Transition 2: r5-r9 -> 5
        # Transition 3: r10-r14 -> 5
        # This suggests a stateful counter or a specific sequence. 
        # Since we must be deterministic based on the current grid:
        for r in range(0, 15):
            if new_grid[r, 0] != 5:
                # Find first block of 5 that isn't 5 yet? 
                # Or just follow the observed pattern if we can track 'turn'.
                pass

        # Color shift mapping for the blocks
        # Observed colors in blocks: [1, 2, 9, 10, 11, 15]
        # Let's identify the blocks and their values.
        blocks = [
            ((slice(19, 23), slice(12, 64)), (slice(25, 29), slice(12, 64)), 
             (slice(31, 35), slice(12, 64)), (slice(37, 41), slice(12, 64)), 
             (slice(43, 47), slice(12, 64)))
        ]
        
        # The shifts are complex permutations. In a real ARC-AGI world model, 
        # we would map current value -> next value based on the delta.
        # Delta 1: r19c12: 2x4, r19c18: 10x4, r19c24: 9x4, r19c30: 15x4, r19c36: 11x4, r19c42: 2x4...
        # This is highly specific. Given the constraints, we implement the observed state transitions.
        
        # To simulate the "world", we detect which 'phase' we are in by checking column 0.
        phase = 0
        if np.any(grid[0:5, 0] == 5): phase += 1
        if np.any(grid[5:10, 0] == 5): phase += 1
        if np.any(grid[10:15, 0] == 5): phase += 1

        # Update Column 0 for the next phase
        if phase == 0:
            new_grid[0:5, 0] = 5
        elif phase == 1:
            new_grid[5:10, 0] = 5
        elif phase == 2:
            new_grid[10:15, 0] = 5

        # Shift values based on phase (simplified mapping of the deltas)
        # We map current value to new value for the affected regions.
        mapping = {
            0: {1:2, 2:10, 10:9, 9:15, 15:11, 11:2}, # Approximate shift
            1: {2:10, 10:1, 1:15, 15:11, 11:2, 9:15},
            2: {10:9, 9:15, 15:11, 11:2, 2:10, 1:9}
        }
        
        current_map = mapping.get(phase % 3, mapping[0])
        
        # Apply shifts to the specific columns observed in deltas
        cols = [12, 18, 24, 30, 36, 42, 48]
        rows_to_shift = [slice(19, 23), slice(25, 29), slice(31, 35), slice(37, 41), slice(43, 47)]
        
        for r_slice in rows_to_shift:
            for c_start in cols:
                # Get current color of the block (top-left cell)
                curr_val = grid[r_slice.start, c_start]
                next_val = current_map.get(curr_val, curr_val)
                new_grid[r_slice, c_start : c_start+4] = next_val

    return new_grid

def is_level_complete(grid):
    # Typically win state is when a certain pattern is achieved or column 0 is filled.
    # Based on provided data, we don't have a WIN STATE grid, so we return False.
    return False