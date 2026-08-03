import numpy as np

def engine(grid, action, data):
    # The observed transitions show a pattern of shifting colors in specific 4x4 blocks.
    # Based on the same ACTION0 repeated multiple times, and the changes occurring in 
    # synchronized shifts across different rows/cols, it seems like a "rotation" or 
    # "shift" of values within these structured regions.
    # However, since only ACTION0 is provided and it's not a coordinate-based click,
    # we must induce a rule that maps the current state to the next state based on the 
    # sequence of observed deltas.
    
    # Let's analyze the shift patterns:
    # In row 19-22, cols 12-15, 18-21, 24-27, 30-33, 36-39, 42-45, 48-51.
    # Initial: [1, 10, 9, 15, 11, 2, 15] (approx)
    # T1: [2, 10, 9, 15, 11, 2, 15] -> wait, let's look at the delta.
    # r19c12: 2x4, r19c18: 10x4...
    # The blocks are shifting right by one position in their list of active slots.
    # Slots: c12, c18, c24, c30, c36, c42, c48.
    # Values at these slots for rows 19-22:
    # Init: c12=1, c18=10, c24=9, c30=15, c36=11, c42=2, c48=15 (Wait, INITIAL grid says r19c12 is not explicitly listed as a block start but implied).
    # Let's re-examine INITIAL grid r19: 14x1, 4x11, 1x4, 4x2, 2x4, 4x2, 10x4, 4x2, 9x4, 4x2, 15x4, 4x2, 11x4, 4x2, 2x4, 4x12.
    # Col indices for values: c12(1), c18(2), c24(10), c30(9), c36(15), c42(11), c48(2).
    # T1: r19c12: 2x4, r19c18: 10x4, r19c24: 9x4, r19c30: 15x4, r19c36: 11x4, r19c42: 2x4, r19c48: 15x4 (Wait, the delta says r19c48 is 15x4).
    # This looks like a cyclic shift of colors within these blocks across the grid.

    if action == 0:
        # Define the block coordinates (top-left corners)
        blocks = [
            (19, 12), (19, 18), (19, 24), (19, 30), (19, 36), (19, 42), (19, 48),
            (20, 12), (20, 18), (20, 24), (20, 30), (20, 36), (20, 42), (20, 48),
            (21, 12), (21, 18), (21, 21), (21, 30), (21, 36), (21, 42), (21, 48), # typo in my analysis
            (22, 12), (22, 18), (22, 24), (22, 30), (22, 36), (22, 42), (22, 48),
            (25, 12), (25, 48),
            (26, 12), (26, 48),
            (27, 12), (27, 48),
            (28, 12), (28, 48),
            (31, 12), (31, 48),
            (32, 12), (32, 48),
            (33, 12), (33, 48),
            (34, 12), (34, 48),
            (37, 12), (37, 48),
            (38, 12), (38, 48),
            (39, 12), (39, 48),
            (40, 12), (40, 48),
            (43, 12), (43, 18), (43, 24), (43, 36), (43, 42), (43, 48),
            (44, 12), (44, 18), (44, 24), (43, 36), (43, 42), (43, 48), # typo
        ]
        # Since we don't have a clear rule for the "win" state or other actions, and ACTION0 is just shifting colors,
        # we will implement a simple cyclic shift of values in these blocks.
        # However, given the constraints and the limited data, the most robust way to handle this is to simulate the observed shifts.
        
        new_grid = grid.copy()
        
        # The pattern: r0-r4 c0=5, then r5-r9 c0=5... it seems like a cursor moving down.
        for r in range(0, 5): new_grid[r, 0] = 5
        # This part is actually happening across transitions.
        # Transition 1: r0-r4 c0=5
        # Transition 2: r5-r9 c0=5
        # Transition 3: r10-r14 c0=5
        # We can track the transition count using a value in the grid if possible, but engine must be pure.
        # Let's assume action 0 triggers the next set of 5 rows at col 0 to become color 5.
        # We need a state variable. Since we cannot have one, we can check how many sets of 5 are already 5.
        count = 0
        for r in range(0, 64):
            if grid[r, 0] == 5:
                count += 1
        sets = count // 5
        start_row = sets * 5
        for r in range(start_row, start_row + 5):
            if r < 64:
                new_grid[r, 0] = 5
        
        # Now for the blocks. The observed deltas show that values shift right.
        # For example, r19c12 becomes 2x4, then 10x4, then 9x4...
        # These colors [2, 10, 9, 15, 11, 2, 15] are shifting.
        # In Transition 1: r19c12=2, r19c18=10, r19c24=9, r19c30=15, r19c36=11, r19c42=2, r19c48=15
        # In Transition 2: r19c12=10, r19c18=9, r19c24=15, r19c30=11, r19c36=2, r19c42=15, r19c48=9
        # In Transition 3: r19c12=9, r19c18=15, r19c24=11, r19c30=2, r19c36=15, r19c42=9, r19c48=10
        # This is a cyclic shift of the sequence [2, 10, 9, 15, 11, 2, 15] (approx).
        # Let's implement this by shifting values in these blocks.
        
        # Define block regions to be shifted
        regions = [
            (slice(19, 23), slice(12, 16)), (slice(19, 23), slice(18, 22)), (slice(19, 23), slice(24, 28)), 
            (slice(19, 23), slice(30, 34)), (slice(19, 23), slice(36, 40)), (slice(19, 23), slice(42, 46)), (slice(19, 23), slice(48, 52)),
            (slice(25, 29), slice(12, 16)), (slice(25, 29), slice(48, 52)),
            (slice(31, 35), slice(12, 16)), (slice(31, 35), slice(48, 52)),
            (slice(37, 41), slice(12, 16)), (slice(37, 41), slice(48, 52)),
            (slice(43, 47), slice(12, 16)), (slice(43, 47), slice(18, 22)), (slice(43, 47), slice(24, 28)), 
            (slice(43, 47), slice(36, 40)), (slice(43, 47), slice(42, 46)), (slice(43, 47), slice(48, 52))
        ]
        
        # For each region group (e.g., the row-like groups), we find the current values and shift them.
        # We'll handle this by shifting all blocks in a "row" of blocks.
        # Row 1: r19-22
        vals1 = [grid[19, 12], grid[19, 18], grid[19, 24], grid[19, 30], grid[19, 36], grid[19, 42], grid[19, 48]]
        new_vals1 = vals1[1:] + [vals1[0]] # Shift left
        for i, col in enumerate([12, 18, 24, 30, 36, 42, 48]):
            new_grid[19:23, col:col+4] = new_vals1[i]
            
        # Row 2: r25-28
        vals2 = [grid[25, 12], grid[25, 48]]
        new_vals2 = vals2[1:] + [vals2[0]]
        for i, col in enumerate([12, 48]):
            new_grid[25:29, col:col+4] = new_vals2[i]
            
        # Row 3: r31-34
        vals3 = [grid[31, 12], grid[31, 48]]
        new_vals3 = vals3[1:] + [vals3[0]]
        for i, col in enumerate([12, 48]):
            new_grid[31:35, col:col+4] = new_vals3[i]
            
        # Row 4: r37-40
        vals4 = [grid[37, 12], grid[37, 48]]
        new_vals4 = vals4[1:] + [vals4[0]]
        for i, col in enumerate([12, 48]):
            new_vals4_val = new_vals4[i]
            new_grid[37:41, col:col+4] = new_vals4_val
            
        # Row 5: r43-46
        vals5 = [grid[43, 12], grid[43, 18], grid[43, 24], grid[43, 36], grid[43, 42], grid[43, 48]]
        new_vals5 = vals5[1:] + [vals5[0]]
        for i, col in enumerate([12, 18, 24, 36, 42, 48]):
            new_grid[43:47, col:col+4] = new_vals5[i]

        return new_grid

    return grid

def is_level_complete(grid):
    # No win state provided. Assume level complete if all rows at col 0 are color 5.
    count = 0
    for r in range(0, 64):
        if grid[r, 0] == 5:
            count += 1
    return count >= 60 # Approximate target (e.g., 60/64)