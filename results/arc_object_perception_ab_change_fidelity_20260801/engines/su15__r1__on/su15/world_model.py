import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action != 6:
        return grid.copy()
    
    px, py = data['x'], data['y']
    new_grid = grid.copy()
    
    # The game seems to be a "capture" or "swap" mechanic where clicking a cell
    # creates a 3x3 square of color 15 at the click location and replaces an existing
    # 3x3 square of color 15 elsewhere.
    # Find all 3x3 blocks of color 15
    blocks_15 = []
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            if np.all(grid[r:r+3, c:c+3] == 15):
                blocks_15.append((r, c))
            # To avoid overlapping detections if we onlys just found one block
            # if np.all(grid[r:r+3, c:c+3] == 15):
                # pass # wait-in-step
    
    # Remove current 3x3 blocks of color 15
    for r, c in blocks_15:
        # We need to a way to handle overlap. If we're not careful, we<|channel>thought
        # The observed deltas show that when ACTION6 is clicked, a new 3x3 block of 15s
        # 
        # Let's refine the search for 3x3 blocks of 15s.
        pass

    # A simpler approach based on the observed transitions:
    # Clicking at (px, py) creates a 3x3 square of color 15 centered at (py, px).
    # In the provided data, clicks are (10, 53), (16, 47), (22, 41), (28, 35), (34, 29).
    # These correspond to coordinates where existing objects were.
    # Clicks are (x=10, y=53), then (x=16, y=47), etc.
    # This means clicking (px, py) places a 3x3 block of 15s starting at row py-2 and col px-2?
    # No, let's look at the delta: r52c9:15x3 r53c9:15x3 r54c9:15x3 for click (10, 53).
    # That is rows 52, 53, 54 and cols 9, 10, 11.
    # Center is (53, 10). So it's a 3x3 block centered at (py, px).
    
    # The observed deltas also show that when a new block is created, an old one is removed.
    # Let's find all 3x3 blocks of color 15 in the current grid.
    blocks_15 = []
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            if np.all(grid[r:r+3, c:c+3] == 15):
                blocks_15.append((r, c))
    
    # To avoid overlapping detections of the same block, we only take unique ones.
    # Since they are 3x3, any overlap would be smaller than 3x3.
    # if len(blocks_15) > <|channel>thought
    # The observed transitions show exactly one 3x3 block of 15s exists at a time.
    # If multiple exist, we remove the "oldest" or some specific one.
    # In these examples, there's always one 3x3 block of 15s.
    
    # Find and remove existing 3x3 block of 15s.
    found_block = False
    for r, c in blocks_15:
        # Check if it's not the one we just created (though we do this after).
        # We use a simple heuristic: the first one found is removed.
        new_grid[r:r+3, c:c+3] = grid[r:r+3, c:c+3] # This doesn't change anything.
        # Wait, what color does it return to?
        # Looking at deltas: r58c3:5x3 r59c3:5x3 r60c3:5x3 for click (10, 53).
        # These were rows 58-60, cols 3-5. That was the INITIAL block of 15s.
        # It returned to color 5.
        new_grid[r:r+3, c:c+3] = 5
        found_block = True
        break

    # Place new 3x3 block of 15s centered at (py, px).
    # The delta says r52c9:15x3... for click (10, 53).
    # So row start = py - 1, col start = px - 1.
    rs, cs = py - 1, px - 1
    if 0 <= rs < grid.shape[0]-2 and 0 <= cs < grid.shape[1]-2:
        new_grid[rs:rs+3, cs:cs+3] = 15
    
    # There's also a change in r63 (the bottom row).
    # Click (10, 53) -> r63c62:5x2
    # Click (16, 47) -> r63c60:5x2
    # Click (22, 41) -> r63c58:5x2
    # This looks like a progress bar or marker moving left.
    # Let's see the pattern: x=10 -> c62; x=16 -> c60; x=22 -> c58; x=28 -> c56; x=34 -> c54.
    # The column is 62 - (px-10)//6 * 2? No.
    # For px=10, col=62. For px=16, col=60. For px=22, col=58.
    # Col = 62 - (px - 10) // 3? No.
    # Difference in px is 6, difference in col is 2.
    # col = 62 - (px - 10) * 2 / 6 = 62 - (px - 10) / 3.
    # Let's check: px=16 -> 62 - 6/3 = 60. Correct.
    # px=22 -> 62 - 12/3 = 58. Correct.
    # px=28 -> 62 - 18/3 = 56. Correct.
    # px=34 -> 62 - 24/3 = 54. Correct.
    # So r63[col : col+2] = 5. But it was already something else.
    # INITIAL grid r63 is 0x64. So it changes from 0 to 5.
    target_col = 62 - (px - 10) // 3
    if 0 <= target_col < grid.shape[1]-1:
        new_grid[63, target_col:target_col+2] = 5

    return new_grid

def is_level_complete(grid):
    # The win state isn't explicitly given, but usually it involves moving the marker or clearing objects.
    # In this case, let's assume it's complete if the block of 15s reaches a certain position.
    # Or simply return False as we don't have a WIN STATE grid.
    return False