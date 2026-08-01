import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where certain blocks (colored squares)
    # move in a pattern based on an action.
    # Based on the observed transitions, ACTION0 is used multiple times.
    # ACTION0 triggers a shift or rotation of the colored blocks within their own group.
    # Let's identify the positions of all "small" colored blocks (the ones that aren't background colors).
    # Background colors are likely 3, 4, 14.
    # Small blocks are colors [1, 2, 5, 8, 9, 10, 11, 15].
    
    if action != 0:
        return grid.copy()

    new_grid = grid.copy()
    
    # Identify all small blocks. A small block is defined by its connected component.
    # We need to find the coordinates of the<|channel>thought
    # and the values of these blocks.
    # Find all cells that are not background colors {3, 4, 14}.
    # a = np.where((grid != 3) & (grid != 4) & (grid != 14))
    # The same logic applies to thedots at the top row.
    # However, looking at the deltas, it's more like a permutation of existing blocks.
    # Let's look at the specific rows and columns where changes occur.
    # Rows: 19-22, 25-28, 31-34, 37-40, 43-46.
    # These are groups of 4x4 blocks.
    # There are also some dots in the first column (r0c0...r14c0).
    # Looking at the ACTION0 transitions:
    # Transition 1: r0c0:5x1 ... r19c12:2x4 etc.
    # Transition 2: r5c0:5x1 ... r19c12:10x4 etc.
    # Transition 3: r10c0:5x1 ... r19c12:9x4 etc.
    # This suggests that the "cursor" or "active block" moves down the left column in steps of 5.
    # And when it does, the colored blocks on the right shift their colors/positions.

    # To implement this simply, we find all 4x4 blocks and rotate their values.
    # Let's identify the top-left corners of these 4x4 blocks.
    # A 4x4 block is a region where grid[y:y+4, x:x+4] has a single color (excluding background).
    # We need to be careful about what constitutes a "block".
    
    blocks = []
    # Scan for 4x4 regions that are not background {3, 4, 14}
    for y in range(0, 64 - 3):
        for x in range(0, 64 - 3):
            region = grid[y:y+4, x:x+4]
            if np.all(region == region[0, 0]) and region[0, 0] not in [3, 4, 14]:
                blocks.append((y, x, region[0, 0]))
                # Skip ahead to avoid overlapping detections of the same block
                # But since they are exactly 4x4 and spaced, it might be okay.

    # The ACTION0 seems to shift colors among these blocks.
    # Let's look at the deltas again. r19c12:2x4 means the block at (19, 12) becomes color 2.
    # In INITIAL GRID, (19, 12) was color 1.
    # This is a permutation.
    # Let's find all such 4x4 blocks and their current colors.
    # We only care about those that actually change.
    # These are located at specific coordinates.
    # Y coords: [19, 25, 31, 37, 43]
    # X coords: [12, 18, 24, 30, 36, 42, 48]
    
    coords = []
    for y in [19, 25, 31, 37, 43]:
        for x in [12, 18, 24, 30, 36, 42, 48]:
            coords.append((y, x))

    # Get current colors of these blocks
    current_colors = [grid[y, x] for y, x in coords]
    
    # The ACTION0 transition seems to be a cyclic shift of the colors among these positions.
    # Transition 1: r19c12 becomes 2 (was 1), r19c18 becomes 10 (was 2)...
    # This is a simple rotation of the `current_colors` list.
    # Let's try rotating by 1 or some fixed amount.
    # In Trans 1: block at (19, 12) was color 1, now it's 2. Block at (19, 18) was 2, now 10.
    # Colors are shifted one position forward in the `coords` list.
    
    shifted_colors = np.roll(current_colors, -1)
    
    new_grid = grid.copy()
    for i, (y, x) in enumerate(coords):
        new_grid[y:y+4, x:x+4] = shifted_colors[i]
        
    # Also handle the left column "cursor" movement.
    # It moves from r0-r4 -> r5-r9 -> r10-r14.
    # The cursor is color 5? No, looking at INITIAL GRID, r0c0 is 14.
    # ACTION0 Transition 1: r0c0 to r4c0 become 5.
    # Transition 2: r5c0 to r9c0 become 5.
    # Transition 3: r10c0 to r14c0 become 5.
    # This means the blocks of 5 cells move down by 5 each time.
    
    # Find where the current '5's are in the first column.
    cursor_pos = -1
    for y in range(0, 15, 5):
        if np.any(grid[y:y+5, 0] == 5):
            cursor_pos = y
            break
    
    # If no 5s found (like in initial grid), start at 0.
    if cursor_pos == -1:
        cursor_pos = -5 # So that next pos is 0
    else:
        cursor_pos += 5
    
    # Clear old cursor and set new one.
    # First, find all existing 5s in col 0 and clear them back to 14.
    new_grid[0:15, 0][new_grid[0:15, 0] == 5] = 14
    if 0 <= cursor_pos < 15:
        new_grid[cursor_pos : cursor_pos + 5, 0] = 5

    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it's when a certain pattern is achieved.
    # Given the data, we can't induce a specific win condition.
    # Return False by default unless a known win state is detected.
    return False