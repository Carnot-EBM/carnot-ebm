import numpy as np

def engine(grid, action, data):
    # The observed transitions show that ACTION0 causes a<|channel>thought
    # shifting or rotating of specific color blocks within the grid.
    # Looking at the delta changes, it's clear that colors are cycling through 
    # positions in a set of defined regions (blocks).
    # Let's identify the "block" coordinates based on the INITIAL GRID and DELTAS.
    # All changes occur in columns 12-15, 18-21, 24-27, 30-33, 36-39, 42-45, 48-51.
    # These are 4x4 blocks starting at x=12, 18, 24, 30, 36, 42, 48.
    # Rows for these blocks are grouped: 19-22, 25-28, 31-34, 37-40, 43-46.
    # Also there's a change to column 0 (color 14 -> 5) in chunks of 5 rows.
    # This suggests a state machine where each action rotates the values of these blocks.

    if action == 0:
        # Define the block centers/top-lefts
        cols = [12, 18, 24, 30, 36, 42, 48]
        rows_groups = [[19, 20, 21, 22], [25, 26, 27, 28], [31, 32, 33, 34], [37, 38, 39, 40], [43, 46]] # Simplified
        # Actually, let's just find all 4x4 blocks that changed.
        # The deltas show changes at r19c12, r19c18... etc.
        # Let's identify all unique top-left corners of 4x4 blocks that are not color 4 (background).
        # We can detect them by looking for non-4 cells and grouping them into 4x4s.
        
        new_grid = grid.copy()
        
        # Shift column 0 colors in chunks of 5 rows
        # Find which chunk is being updated based on current state
        chunk_size = 5
        for i in range(0, 64, chunk_size):
            if i + chunk_size <= 64:
                # Check if this chunk is already color 5
                already_done = True
                for r in range(i, i+chunk_size):
                    if grid[r, 0] != 5:
                        already_done = False
                        break
                if not already_done:
                    # This is the same as ACTION0 logic observed: it updates one chunk per call?
                    # No, the observations show different chunks updating in sequence.
                    # But we need a deterministic engine. 
                    # Since the prompt asks for simple general rules, let's see:
                    # Action 0 seems to cycle values among these 4x4 blocks.
                    pass

        # The most consistent pattern is that values are rotating across the block positions.
        # Let's define the fixed set of 4x4 block top-left corners.
        block_tl = []
        for r in [19, 25, 31, 37, 43]:
            for c in [12, 18, 24, 30, 36, 42, 48]:
                block_tl.append((r, c))
        
        # For each row group (e.g., rows 19-22), colors shift horizontally.
        # For other groups, they might shift differently.
        # Looking at r19c12: 2x4 -> 10x4 -> 9x4 ... this is a rotation of values.
        # Values involved: [2, 10, 9, 15, 11, 2, 15] etc.
        
        # Because we must be deterministic and the observed ACTION0 transitions 
        # happen sequentially on different parts of the grid, it implies there's an internal state.
        # However, engine() must be pure. This suggests action 0 always does the same thing to the current grid.
        
        # Let's implement the most obvious part: shifting block values.
        # We identify all 4x4 blocks, extract their values, rotate them, and put them back.
        
        blocks = [] # list of (r, c, value)
        for r in [19, 25, 31, 37, 43]:
            for c in [12, 18, 24, 30, 36, 42, 48]:
                val = grid[r, c]
                if val != 4:
                    blocks.append([r, c, val])
        
        # The deltas show a complex rotation. A simple shift might not match perfectly,
        # but for ARC-AGI world models, often "rotate colors" is the key.
        # Shift colors by one position in the identified sequence.
        if len(blocks) > 0:
            vals = [b[2] for b in blocks]
            rotated_vals = vals[1:] + [vals[0]]
            for i in range(len(blocks)):
                new_grid[blocks[i][0]:blocks[i][0]+4, blocks[i][1]:blocks[i][1]+4] = rotated_vals[i]

        # Update column 0 chunks
        # Find first chunk of 5 that isn't color 5
        for i in range(0, 64, 5):
            if i+5 <= 64 and np.all(grid[i:i+5, 0] != 5):
                new_grid[i:i+5, 0] = 5
                break
                
        return new_grid

    return grid

def is_level_complete(grid):
    # Level complete usually means a certain pattern is achieved or all targets are collected.
    # In this case, maybe when all column 0 cells (up to some point) are color 5?
    # Or when the block colors reach a specific configuration.
    # Given the data, we don't have a WIN STATE grid, so we guess based on common ARC patterns.
    return np.all(grid[0:30, 0] == 5) # Example condition