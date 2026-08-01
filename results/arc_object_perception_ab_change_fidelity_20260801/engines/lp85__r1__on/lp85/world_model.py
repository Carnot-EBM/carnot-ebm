import numpy as np

def engine(grid, action, data):
    # The game appears to be a puzzle where certain blocks (the "colored squares")
    # move in a coordinated way based on an action.
    # In the observed transitions, ACTION0 is used repeatedly.
    # Since only one action is provided and it's not a coordinate-based click,
    # we assume it's a trigger for a state transition.
    
    # Based on the<|channel>thought process, the colored blocks shift their positions.
    # Specifically, they seem to rotate or cycle through a set of predefined slots.
    # Let's identify the slots.
    
    # Slots are 4x4 areas.
    slots = []
    # Row indices for slots (approximate from initial grid)
    row_groups = [19, 25, 31, 37, 43] # These are actually y coordinates
    col_groups = [12, 18, 24, 30, 36, 42, 48] # x coordinates
    
    # We need to find all existing colored blocks (excluding background colors 3, 4, 14).
    # Background colors: 3, 4, 14.
    bg_colors = {3, 4, 14}
    
    # Find current block positions
    blocks = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] not in bg_colors:
                # This is part of a block. Since blocks are 4x4, we can group them.
                # If it's not already processed, mark as visited.
                pass

    # Instead of let's use the object structure provided in the transitions.
    # The blocks move between specific slots.
    # Let's define the slot centers or top-left corners.
    # From INITIAL GRID and ACTION0 deltas:
    # Slots at (y, x):
    # Group 1: y=19, x=[12, 18, 24, 30, 36, 42, 48]
    # Group 2: y=25, x=[12, 48]
    # Group 3: y=31, x=[12, 48]
    # Group 4: y=37, x=[12, 48]
    # Group 5: y=43, x=[12, 18, 24, 30, 36, 42, 48]
    
    # Looking at the deltas, they shift colors across these positions.
    # It looks like a permutation of the colored blocks.
    
    # Since we only have one action and it cycles states, let's implement a simple cycle.
    # We identify all unique "block" colors present in the grid.
    # The block colors are [1, 2, 9, 10, 11, 15].
    # Let's find where these colors are currently located (top-left corners).
    
    block_colors = {1, 2, 9, 10, 11, 15}
    current_blocks = [] # List of (y, x, color)
    visited = np.zeros_like(grid, dtype=bool)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] in block_colors and not visited[r, c]:
                color = grid[r, c]
                # Find bounds of this block
                # Assuming blocks are roughly square/rectangular
                # For simplicity, just take the first pixel as top-left
                current_blocks.append((r, c, color))
                # Mark the rest of the block as visited to avoid duplicates
                # Blocks seem to be 4x4 or similar
                visited[r:r+4, c:c+4] = True

    # The ACTION0 transitions show a shift.
    # In the first transition, colors at y=19 move right? Or cycle?
    # Let's look at y=19: [1, 2, 10, 9, 15, 11, 2] -> [2, 10, 9, 15, 11, 2, 15]? No.
    # Actually, let's observe the deltas for y=19 specifically:
    # Initial: r19c12:1x4, r19c18:2x4, r19c24:10x4, r19c30:9x4, r19c36:15x4, r19c42:11x4, r19c48:2x4
    # Delta 1: r19c12:2x4, r19c18:10x4, r19c24:9x4, r19c30:15x4, r19c36:11x4, r19c42:2x4, r19c48:15x4
    # This is a shift to the left! (The color at c18 moves to c12, etc.)
    # Wait: 2->12, 10->18, 9->24, 15->30, 11->36, 2->42, 15->48.
    # The sequence [1, 2, 10, 9, 15, 11, 2] becomes [2, 10, 9, 15, 11, 2, 15].
    # It's a rotation of the colors present in those slots.
    
    # Let's implement this as a general "shift" for all block-containing rows.
    new_grid = grid.copy()
    
    # Identify rows that contain blocks and their column positions
    block_rows = [19, 20, 21, 22, 25, 26, 27, 28, 31, 32, 33, 34, 37, 38, 39, 40, 43, 44, 45, 46]
    col_slots = [12, 18, 24, 30, 36, 42, 48]
    
    for r in block_rows:
        # Get current colors in the slots for this row
        colors = [grid[r, c] for c in col_slots]
        # Shift left (rotate)
        shifted_colors = colors[1:] + [colors[0]]
        # Apply shifted colors back to the grid
        for c, color in zip(col_slots, shifted_colors):
            new_grid[r, c] = color

    # Also handle the "side" objects like the ones at c=0.
    # Initial: r0c0:14x1... Delta 1: r0c0:5x1. Color 14 becomes 5.
    # This looks like a separate cycle of colors on the far left column.
    left_col_colors = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
    # The deltas show r0-r4 becoming color 5, then r5-r9 becoming color 5...
    # This is a moving window of color 5.
    
    # Let's find the current "window" of color 5 in column 0.
    win_start = -1
    for r in range(64):
        if grid[r, 0] == 5:
            win_start = r
            break
    
    if win_start == -1: # Initial state
        new_grid[0:5, 0] = 5
    else:
        # Move window down by 5
        new_grid[win_start : win_start+5, 0] = 14 # Reset old
        new_start = (win_start + 5) % 64
        # Handle wrap around for the window
        for i in range(5):
            new_grid[(new_start + i) % 64, 0] = 5

    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it involves aligning colors or clearing blocks.
    # Given the data, we can't be sure. Return False as default.
    return False