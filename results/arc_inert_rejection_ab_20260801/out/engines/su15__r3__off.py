import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # Extract click coordinates
    px, py = data['x'], data['y']
    
    # The game seems to be a puzzle where clicking on certain areas triggers changes.
    # Based on the observed transitions, clicking at specific points moves a 3x3 block of color 15.
    # A 3x3 block of color 15 exists initially at (4, 30), (5, 30), (6, 30).
    # There are also blocks of color 15 at (58, 3), (59, 3), (60, 3).
    #<|channel>thought
    # Looking at the same pattern:
    # ACTION6 x=10, y=53 -> r52c9:15x3, r53c9:15x3, r54c9:15x3 AND r58c3:5x3, r59c3:5x3, r60c3:5x3
    # ACTION6 x=16, y=47 -> r46c15:15x3, r47c15:15x3, r48c15:15x3 AND r52c9:5x3, r53c9:5x1,3x1,5x1, r54c9:5x3
    # The click coordinate (px, py) is exactly the center of the new 3x3 block.
    # The 3x3 block of color 15 moves to be centered at (py-1, px-1)? No, let's check.
    # Click (10, 53): Block at rows 52, 53, 54 and cols 9, 10, 11. Center is (53, 10). Correct.
    # Click (16, 47): Block at rows 46, 47, 48 and cols 15, 16, 17. Center is (47, 16). Correct.
    # Click (22, 41): Block at rows 40, 41, 42 and cols 21, 22, 23. Center is (41, 22). Correct.
    # Click (28, 35): Block at rows 34, 35, 36 and cols 27, 28, 29. Center is (35, 28). Correct.
    # Click (34, 29): Block at rows 28, 29, 30 and cols 33, 34, 35. Center is (29, 34). Correct.
    
    # The block of color 15 moves to the click location.
    # Additionally, it seems that clicking a point "clears" or "collects" something.
    # Note the changes in r63: r63c62:5x2 -> r63c60:5x2 -> r63c58:5x2...
    # This looks like a progress bar or counter.
    
    new_grid = grid.copy()
    
    # Move the 3x3 block of color 15 to be centered at (py, px)
    # First, find all existing 3x3 blocks of color 15 and clear them? 
    # Or just move one specific block?
    # In the transitions, only one 3x3 block of color 15 exists at a time (except for initial state).
    # Let's assume we replace any 3x3 area of color 15 with background color 5 (or whatever was there).
    # Based on the deltas, the previous 3x3 block becomes color 5 (mostly).
    
    # Find current 3x3 block of color 15
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            if np.all(grid[r:r+3, c:c+3] == 15):
                new_grid[r:r+3, c:c+3] = 5 # Clear it
                break
        else: continue
        break

    # Place new 3x3 block of color 15 at click location
    # Ensure coordinates are within bounds
    r_start, c_start = py - 1, px - 1
    if 0 <= r_start < grid.shape[0]-2 and 0 <= c_start < grid.shape[1]-2:
        new_grid[r_start:r_start+3, c_start:c_start+3] = 15
    
    # The progress bar at r63 changes by 2 pixels each time.
    # Let's find the current "gap" or "marker" in r63 and move it.
    # In INITIAL: r63 is all 0.
    # After Action 1: r63c62:5x2 (cols 62, 63 become 5)
    # After Action 2: r63c60:5x2 (cols 60, 61 become 5)
    # This means we are filling from right to left? No, the delta says "changed cells".
    # If r63 was all 0, then r63c62:5x2 makes cols 62, 63 color 5.
    # Then r63c60:5x2 makes cols 60, 61 color 5.
    # So it's a growing line of color 5 starting from the right.
    
    # Find how many color 5 cells are already at the end of r63
    count_5 = 0
    for c in range(grid.shape[1]-1, -1, -1):
        if grid[63, c] == 5:
            count_5 += 1
        else:
            break
    
    new_grid[63, grid.shape[1] - count_5 - 2 : grid.shape[1] - count_5] = 5
    
    return new_grid

def is_level_complete(grid):
    # Level complete when the progress bar (r63) is full or some condition is met.
    # Since we don't have a win state, let's assume it's when the block reaches a target or r63 is full.
    return np.all(grid[63, :] == 5)