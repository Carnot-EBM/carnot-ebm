import numpy as np

def engine(grid, action, data):
    # The observed transitions show a pattern of shifting colors in specific blocks.
    # Action 0 seems to trigger these shifts.
    # Based on the delta, it's a cyclic shift of colors within certain regions.
    # Let's identify the same-colored cells that are part of this "shifting" mechanism.
    # grid shape is (64, 64)
    
    if action != 0:
        return grid.copy()
    
    new_grid = grid.copy()
    
    # Identify all unique colors present in the grid except for background/border colors
    # We assume border color is 14 or 3 or 4 based on INITIAL GRID.
    # These are likely static elements.
    #<|channel>thought
    # In ACTION0, we actually see changes in column 0 and columns 12, 18, 24, 30, 36, 42, 48.
    # This suggests a set of vertical strips or blocks.
    # The values being shifted are [1, 2, 9, 10, 11, 15].
    # These appear as 4x4 blocks.
    # The x-coordinates of these blocks are 12, 18, 24, 30, 36, 42, 48.
    # The y-coordinates of these blocks are 19, 25, 31, 37, 43.
    # Let's define the block coordinates.
    block_cols = [12, 18, 24, 30, 36, 42, 48]
    block_rows = [19, 25, 31, 37, 43]
    
    # Create a list of all current colors in these blocks
    blocks = []
    for r in block_rows:
        for c in block_cols:
            val = grid[r, c]
            blocks.append(val)
    
    # The observed ACTION0 transitions show that every time action 0 is pressed,
    # the colors shift. Specifically, it looks like they might be shifting by one position
    # in some order.
    # Looking at the first transition:
    # r19c12 becomes 2, r19c18 becomes 10, r19c24 becomes 9...
    # In the second transition:
    # r19c12 becomes 10, r19c18 becomes 9, r19c24 becomes 15...
    # This is a cyclic shift of the values across the blocks.
    
    # Let's find the sequence of colors being shifted.
    # From INITIAL GRID:
    # r19c12=1, r19c18=10, r19c24=9, r19c30=15, r19c36=11, r19c42=2, r19c48=15 (Wait, this isn't quite right).
    # Let's re-examine the delta for Action 0.
    # Transition 1: r19c12:2x4, r19c18:10x4, r19c24:9x4, r19c30:15x4, r19c36:11x4, r19c42:2x4, r19c48:15x4
    # Wait, the deltas are just replacing the 4x4 blocks with new colors.
    # The pattern is that all 4x4 blocks at these coordinates change their color simultaneously.
    
    # Let's look at the values in a row of blocks (e.g., row 19):
    # Initial: [1, 10, 9, 15, 11, 2, 15] (approx)
    # T1: [2, 10, 9, 15, 11, 2, 15] -> No, let me read carefully.
    # ACTION0 Delta 1: r19c12:2x4, r19c18:10x4, r19c24:9x4, r19c30:15x4, r19c36:11x4, r19c42:2x4, r19c48:15x4
    # This means for rows 19-22 and cols 12-15, value becomes 2. For cols 18-21, value becomes 10...
    
    # Actually, looking at the sequence of Action 0 deltas:
    # The colors are shifting cyclicly across the block positions.
    # Let's define the order of block positions as a list of (r, c).
    positions = []
    for r in block_rows:
        for c in block_cols:
            positions.append((r, c))
            
    # Get current values
    current_values = [grid[r, c] for r, c in positions]
    
    # Shift them by one position (cyclic)
    # Based on the data, it looks like they shift "left" or "right".
    # Let's try a simple rotation.
    new_values = np.roll(current_values, -1)
    
    # Apply to grid
    for i, (r, c) in enumerate(positions):
        new_grid[r:r+4, c:c+4] = new_values[i]
        
    # Also handle column 0 changes seen in ACTION0.
    # r0c0:5x1, r1c0:5x1... these are just changing color 14 to 5.
    # It seems every Action 0 also increments/changes the border color at col 0.
    # Transition 1: rows 0-4 become 5.
    # Transition 2: rows 5-9 become 5.
    # Transition 3: rows 10-14 become 5.
    # This is a sequential filling of the left border with color 5.
    
    # We need to track how many times action 0 has been called.
    # Since we don't have state, we can infer it from the current grid.
    count = 0
    for r in range(64):
        if grid[r, 0] == 5:
            count += 1
    
    start_row = (count // 5) * 5
    # If count is already multiple of 5, and we are adding another block, 
    # we should check if the next block is already 5.
    # But let's assume they are added in blocks of 5.
    next_block_start = (count // 5) * 5
    if next_block_start < 60:
        new_grid[next_block_start : next_block_start + 5, 0] = 5
    # Wait, looking at ACTION0 deltas again:
    # T1: r0c0...r4c0 become 5.
    # T2: r5c0...r9c0 become 5.
    # T3: r10c0...r14c0 become 5.
    # This matches perfectly.
    
    # Correcting the border logic:
    current_filled = 0
    while current_filled < 64 and grid[current_filled, 0] == 5:
        current_filled += 1
    
    # The observed transitions show that for each Action 0, a new block of 5 cells becomes color 5.
    # However, the first transition starts at row 0.
    # If rows 0-4 are already 5, then rows 5-9 should become 5.
    fill_start = 0
    while fill_start < 60 and grid[fill_start, 0] == 5:
        fill_start += 1
    # But wait, if it's exactly 5, 10, etc., we need to be careful.
    # Let's just find the first index that is NOT 5.
    first_not_5 = 0
    while first_not_5 < 64 and grid[first_not_5, 0] == 5:
        first_not_5 += 1
    
    if first_not_5 < 60:
        new_grid[first_not_5 : first_not_5 + 5, 0] = 5
    elif first_not_5 < 64:
        new_grid[first_not_5 : 64, 0] = 5

    return new_grid

def is_level_complete(grid):
    # Level complete usually means a certain pattern is reached or all targets collected.
    # In this case, maybe when the left border is fully color 5?
    # Or some other condition. Given no win state, let's guess based on common ARC patterns.
    # The most likely completion is filling the border.
    return np.all(grid[:60, 0] == 5)