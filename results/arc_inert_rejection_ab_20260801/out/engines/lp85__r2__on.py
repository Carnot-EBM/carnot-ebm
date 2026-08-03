import numpy as np

def engine(grid, action, data):
    # The observed transitions show that ACTION0 (which seems to be a keyboard/directional 
    # action) triggers a cyclic shift of colors in specific regions of the grid.
    # Based on the same-color blocks (e.g., 4x4 blocks),s it appears the game involves shifting 
    # these blocks' values own their positions.
    # Identify all unique color blocks (excluding background and boundary colors).
    # Background is usually 3 or 4.
    #
    # In the initial grid, we see patterns of 4x4 blocks.
    # Let's define the blocks based on the coordinates provided in the run-length encoding.
    # We umare identifying the<|channel>thought process but writing only the function.
    
    if action != 0:
        return grid.copy()
    
    new_grid = grid.copy()
    
    # Define the areas where shifts occur. These are 4x4 blocks.
    # Blocks are located at rows [19,22], [25,28], [31,34], [37,40], [43,46]
    # And columns [12,15], [18,21], [24,27], [30,33], [36,39], [42,45], [48,51]
    
    rows_blocks = [[19, 22], [25, 28], [31, 34], [37, 40], [43, 46]]
    cols_blocks = [[12, 15], [18, 21], [24, 27], [30, 33], [36, 39], [42, 45], [48, 51]]
    
    # The observed transitions show that ACTION0 causes a shift in colors across these blocks.
    # Let's collect all current values of these blocks.
    # For each block (r_start, r_end), (c_start, c_end) and the umare identifying the pattern.
    # In transition 1: 
    # Block at (19,12) changes from 4 to 2, (19,18) from 11 to 10, etc.
    # It looks like a cyclic permutation of the color set {1, 2, 9, 10, 11, 15}.
    # Color mapping for ACTION0:
    # Transition 1: 4->2, 11->10, 4->9, 10->15, 9->11, 15->2... wait.
    # Looking closer at the delta:
    # r19c12: 2x4, r19c18: 10x4, r19c24: 9x4, r19c30: 15x4, r19c36: 11x4, r19c42: 2x4, r19c48: 15x4
    # This is not a simple global map. It's a shift within each row of blocks.
    
    # Let's use the observed deltas directly as they are very consistent.
    # The colors in these blocks cycle through [2, 10, 9, 15, 11] or similar.
    # Based on the three transitions provided:
    # T1: Block(19,12)=2, (19,18)=10, (19,24)=9, (19,30)=15, (19,36)=11, (19,42)=2, (19,48)=15
    # T2: Block(19,12)=10, (19,18)=9, (19,24)=15, (19,30)=11, (19,36)=2, (19,42)=15, (19,48)=9
    # T3: Block(19,12)=9, (19,18)=15, (19,24)=11, (19,30)=2, (19,36)=15, (19,42)=9, (19,48)=10
    # This is a cyclic shift of values across the blocks in each row.
    
    # The sequence for the first 5 blocks in row 19-22: [2, 10, 9, 15, 11]
    # Shift 1: 2, 10, 9, 15, 11
    # Shift 2: 10, 9, 15, 11, 2
    # Shift 3: 9, 15, 11, 2, 10
    # This is exactly it!
    
    # Now we need to apply this to all block rows and columns.
    # Row sets:
    # R1: [19,22], R2: [25,28], R3: [31,34], R4: [37,40], R5: [43,46]
    # Col sets:
    # C1: [12,15], C2: [18,21], C3: [24,27], C4: [30,33], C5: [36,39], C6: [42,45], C7: [48,51]
    
    # The values in these blocks are shifted left by one position each ACTION0.
    # Let's implement a general shift for any 4x4 block grid.
    
    for rs in rows_blocks:
        for cs_idx in range(len(cols_blocks)):
            # Find the value of the current block (top-left cell)
            val = grid[rs[0]][cols_blocks[(cs_idx + 1) % len(cols_blocks)][0]]
            # Set the current block to that value
            new_grid[rs[0]:rs[1]+1, cols_blocks[cs_idx][0]:cols_blocks[cs_idx][1]+1] = val
            
    # Also handle the boundary color change seen in deltas: r0c0:5x1 etc.
    # This looks like a counter or state indicator on the left edge.
    # T1: r0..r4 c0 -> 5; T2: r5..r9 c0 -> 5; T3: r10..r14 c0 -> 5
    # It moves down by 5 cells every action.
    
    # We can determine the current "marker" position and move it.
    # But since we don't have the full history, let's just find where '5' is and shift it.
    current_marker_row = -1
    for r in range(64):
        if grid[r, 0] == 5:
            current_marker_row = r
            break
    
    # If no marker found (initial), start at 0? No, initial has 14.
    # Let's assume if not found, first ACTION0 puts it at 0-4.
    start_row = current_marker_row + 5 if current_marker_row != -1 else 0
    if start_row < 64:
        new_grid[start_row : min(start_row+5, 64), 0] = 5
        # Clear old marker
        if current_marker_row != -1:
            new_grid[current_marker_row : min(current_marker_row+5, 64), 0] = 14

    return new_grid

def is_level_complete(grid):
    # Win state usually involves a specific pattern or all blocks matching.
    # Without a win state example, we return False unless a clear condition is met.
    return False