import numpy as np

def engine(grid, action, data):
    # The observed transitions show a pattern of shifting colors in specific blocks.
    # Action 0 seems to be a "cycle" or "shift" operation that affects multiple areas of the grid.
    # Based on thes deltas, it's a a global shift of values within certain predefined regions.
    # Let's identify the same-sized blocks (4x4) and find how they are part of the cycle.
    # 
    # Looking at the laout, there are several 4x4 blocks of uniform color.
    #<|channel>thought
    # The logic appears to be a simple value rotation among a set of colors own by 4x4 blocks.
    # In each ACTION0, the colors rotate.
    # For example, let's look at the first delta:
    # r19c12: 2x4, r19c18: 10x4... etc.
    # It looks like if a block was color A, it becomes B, then C, then D.
    # We need to map which blocks change together.
    # Find all 4x4 blocks of single color.
    # 
    # The background is mostly 3s and 4s.
    # Blocks are usually colors [1, 2, 5, 8, 9, 10, 11, 15].
    # 
    # Let's implement a general "rotate values" rule for Action 0.
    # Since we only have one action (ACTION0), and it rotates specific sets of blocks,
    # we can assume that any 4x4 block with a color not in {3, 4} will rotate its color.
    # 
    # Looking closely at the deltas:
    # r19c12: 2x4 -> r19c18: 10x4 ... this means the block at c12 became 2, while the one at c18 became 10.
    # This suggests a rotation among the positions of these blocks.
    # 
    # Actually, looking at the ACTION0 transitions again:
    # Transition 1: Block(r19, c12) becomes 2, Block(r19, c18) becomes 10...
    # Transition 2: Block(r19, c12) becomes 10, Block(r19, c18) becomes 9...
    # Transition 3: Block(r19, c12) becomes 9, Block(r19, c18) becomes 15...
    # 
    # It's a value shift: Value(pos_i) = Value(pos_{i-1})
    # Let's find all coordinates (r, c) that are top-left corners of 4x4 blocks of uniform color (excluding background).
    # Then we rotate their values.
    
    if action != 0:
        return grid.copy()

    new_grid = grid.copy()
    blocks = []
    # Scan for 4x4 blocks of single color (not 3 or 4)
    for r in range(0, 64 - 3, 4): # Heuristic step 4 based on block size
        for c in range(0, 64 - 3, 4):
            block = grid[r:r+4, c:c+4]
            if np.all(block == block[0, 0]) and block[0, 0] not in [3, 4]:
                blocks.append((r, c))
    
    # If no blocks found with step 4, try every pixel
    if not blocks:
        for r in range(0, 64 - 3):
            for c in range(0, 64 - 3):
                block = grid[r:r+4, c:c+4]
                if np.all(block == block[0, 0]) and block[0, 0] not in [3, 4]:
                    blocks.append((r, c))
                    # skip to avoid overlapping same block
                    # but we need a unique set of non-overlapping blocks
    
    # To handle the rotation correctly, we must identify all such blocks first.
    # Let's refine the search for blocks to be more robust.
    # The deltas show changes at specific columns like 12, 18, 24, 30... (diff=6)
    # Rows are 19, 25, 31, 37, 43... (diff=6)
    
    # Re-scanning based on observed coordinates
    coords = []
    # Observed rows: 19, 20, 21, 22 | 25, 26, 27, 28 | 31, 32, 33, 34 | 37, 38, 39, 40 | 43, 44, 45, 46
    # Observed cols: 12, 18, 24, 30, 36, 42, 48
    rows_starts = [19, 25, 31, 37, 43]
    cols_starts = [12, 18, 24, 30, 36, 42, 48]
    
    for rs in rows_starts:
        for cs in cols_starts:
            coords.append((rs, cs))
            
    # Now we rotate the values of these blocks.
    # The rotation seems to be a simple shift along the list of coords.
    vals = [grid[r, c] for r, c in coords]
    rotated_vals = np.roll(vals, 1)
    
    for (r, c), v in zip(coords, rotated_vals):
        new_grid[r:r+4, c:c+4] = v
        
    # Also handle the left column change observed in deltas: r0-r4c0: 5x1, etc.
    # This looks like a separate cycle on the first column.
    col0_indices = [0, 5, 10] # Based on r0-r4, r5-r9, r10-r14
    # In transition 1: r0-r4 becomes 5. Transition 2: r5-r9 becomes 5. Transition 3: r10-r14 becomes 5.
    # It's a moving block of color 5.
    # Let's find where the block of 5 is currently.
    current_pos = -1
    for i in range(len(col0_indices)):
        start = col0_indices[i]
        if grid[start, 0] == 5:
            current_pos = i
            break
    
    next_pos = (current_pos + 1) % len(col0_indices)
    # Clear old pos, set new pos
    if current_pos != -1:
        new_grid[col0_indices[current_pos]:col0_indices[current_pos]+5, 0] = grid[col0_indices[current_pos], 0] # This is wrong logic
        # The deltas show only the NEW values.
        # Transition 1: r0-r4c0=5. Transition 2: r5-r9c0=5. Transition 3: r10-r14c0=5.
        # Let's just move the '5' block.
        # First, reset all these segments to a default (e.g., color 14 from initial grid)
        for start in col0_indices:
            new_grid[start:start+5, 0] = 14
        new_grid[col0_indices[next_pos]:col0_indices[next_pos]+5, 0] = 5

    return new_grid

def is_level_complete(grid):
    # No win state provided, assume False or define based on common ARC patterns.
    return False