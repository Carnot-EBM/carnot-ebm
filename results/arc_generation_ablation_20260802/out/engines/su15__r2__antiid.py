import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid
    
    # The game seems to involve clicking on cells (ACTION6)
    # Clicking on a cell changes the state of a 3x3 area around the click point.
    # The same time, it's some kind of 'collection' or 'clearing' mechanism.
    # Looking at the transitions:
    # ACTION6 x=10, y=53 -> r52c9:15x3, r53c9:15x3, r54c9:15x3, r58c3:5x3, r59c3:5x3, r60c3:5x3, r63c62:5x2
    # This matches a 3x3 block starting at (y-1, x-1) if we assume logical coordinates.
    # Let's refine this: the click target is (data['y'], data['x']).
    # The<|channel>thought process is part be able to identify that clicking on a 3x3 block of color 15 (or similar)
    # and changing it to something else, or vice versa.
    # In the first transition, the click was at (53, 10). The cells changed from 0/3 to 15.
    # {r52c9:15x3, r53c9:15x3, r54c9:15x3} are exactly a 3x3 block centered at (53, 10).
    # And {r58c3:5x3, r59c3:5x3, r60c3:5x3} were color 15 and became 5.
    # And {r63c62:5x2} changes some bottom edge pixels.
    # It looks like there is a "cursor" or "active block" of size 3x3 that moves or is teleported.
    # Based on the transitions, each click creates a new 3x3 block of color 15 at the clicked location,
    # and removes the previous 3x3 block of color 15.
    # Also, there's a progress bar at the bottom row (r63) filling up with color 5.
    # Let's implement this behavior.

    new_grid = grid.copy()
    px, py = data['x'], data['y']
    
    # Find existing 3x3 blocks of color 15
    # We look for any cell that is color 15 and part of a 3x3 block.
    # Since we only have one such block moving, let's find its top-left corner.
    prev_block_pos = None
    for r in range(64):
        for c in range(64):
            if grid[r, c] == 15:
                # Check if it's the top-left of a 3x3 block of 15s
                if r + 2 < 64 and c + 2 < 64:
                    if np.all(grid[r:r+3, c:c+3] == 15):
                        prev_block_pos = (r, c)
                        break
        if prev_block_pos: break

    # Remove previous block (set to background color based on context, e.g., 5 or 4)
    if prev_block_pos:
        pr, pc = prev_block_pos
        # The observed transitions show the same area becoming color 5 or similar.
        # Let's use 5 as the default "cleared" color for the main area.
        new_grid[pr:pr+3, pc:pc+3] = 5
        # Special case: if the block was at the very bottom, maybe different?
        # No, let's stick to 5.
    
    # Place new block at clicked location (centered)
    # Click is (px, py). Center it.
    start_r, start_c = py - 1, px - 1
    if 0 <= start_r < 62 and 0 <= start_c < 62:
        new_grid[start_r:start_r+3, start_c:start_c+3] = 15
    
    # Update progress bar at r63
    # Each click seems to fill 2 pixels of color 5 in row 63 from right to left.
    # Row 63 starts as all 0s.
    # Transition 1: x=10, y=53 -> r63c62:5x2 (cols 62, 63 become 5)
    # Transition 2: x=16, y=47 -> r63c60:5x2 (cols 60, 61 become 5)
    # This means the progress bar fills from right to left.
    # Let's find how many are already filled.
    filled_count = np.sum(grid[63, :] == 5)
    next_col_start = 63 - filled_count - 1 # Approximate
    # To be precise: if 0 filled, cols 62,63. If 2 filled, cols 60,61.
    # So it's indices [63-filled-2 : 63-filled] ? No.
    # Trans 1: 0 filled -> c62, c63.
    # Trans 2: 2 filled -> c60, c61.
    # The formula is: start_col = 63 - (filled_count + 2)
    # Wait, let's just fill two pixels to the left of the current leftmost '5'.
    leftmost_five = np.where(grid[63, :] == 5)[0]
    if len(leftmost_five) > 0:
        l_idx = leftmost_five[0]
        new_grid[63, max(0, l_idx-2):max(0, l_idx)] = 5
    else:
        new_grid[63, 62:64] = 5

    return new_grid

def is_level_complete(grid):
    # Level complete when progress bar r63 is full or a certain condition is met.
    # In this game, usually filling the bar is the goal.
    return np.all(grid[63, :] == 5)