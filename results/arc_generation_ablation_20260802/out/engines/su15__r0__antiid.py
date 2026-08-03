import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid
    
    # The game seems to be a puzzle where clicking on a cell (x, y)
    # creates a 3x3 block of color 15 at that location and replaces
    # another 3x3 block of color 15 elsewhere.
    # Therethoughts own logic based on the<|channel>thoughts same coordinates.
    #
    # Based on the observations:
    # ACTION6 x=10, y=53 -> r52c9:15x3, r53c9:15x3, r54c9:15x3 AND r58c3:5x3, r59c3:5x3, r60c3:5x3
    # ACTION6 x=16, y=47 -> r46c15:15x3, r47c15:15x3, r48c15:15x3 AND r52c9:5x3, r53c9:5x1, 3x1, 5x1, r54c9:5x3
    # ACTION6 x=22, y=41 -> r40c21:15x3, r47c15:5x3... (Wait, it's a sequence)
    # It looks like clicking at (x, y) creates a 3x3 block of color 15 centered at (y-1, x-1)? No.
    # Let's look at the coordinates:
    # Click (10, 53): Block at row 52, col 9 (top-left). 52 = 53-1, 9 = 10-1.
    # Click (16, 47): Block at row 46, col 15 (top-left). 46 = 47-1, 15 = 16-1.
    # Click (22, 41): Block at row 40, col 21 (top-left). 40 = 41-1, 21 = 22-1.
    # Click (28, 35): Block at row 34, col 27 (top-left). 34 = 35-1, 27 = 28-1.
    # Click (34, 29): Block at row 28, col 33 (top-left). 28 = 29-1, 33 = 34-1.
    # In each case, a new 3x3 block of color 15 is created at (y-1, x-1) to (y+1, x+1).
    # And the previous 3x3 block that was "active" or "special" is replaced by color 5.
    # The first click (10, 53) replaces the block at r58c3:r60c5 (which was color 15 in INITIAL GRID).
    # The second click (16, 47) replaces the block at r52c9:r54c11.
    # The third click (22, 41) replaces the block at r46c15:r48c17.
    # This is a sequence where clicking moves the "highlighted" 3x3 block.
    
    new_grid = grid.copy()
    px, py = data['x'], data['y']
    
    # Create new 3x3 block of color 15
    start_row, start_col = py - 1, px - 1
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if 0 <= r < 64 and 0 <= c < 64:
                new_grid[r, c] = 15
    
    # Find the existing 3x3 block of color 15 that should be removed
    # In this specific game, it seems to be the one that was created by the previous action or the initial one.
    # We search for any 3x3 block of color 15 and replace it with color 5.
    # However, there's also a change at r63 (the bottom row).
    # Let's look at the delta: r63c62:5x2, then r63c60:5x2... it's moving left.
    
    # To implement the "remove old block" rule simply:
    # find all 3x3 blocks of color 15 and remove them before adding the new one?
    # No, because we need to keep the state.
    # The observed deltas show only ONE 3x3 block of color 15 exists at a time in those regions.
    
    # Search for current 3x3 block of color 15 and turn it into color 5
    for r in range(64):
        for c in range(64 - 2):
            if np.all(grid[r:r+3, c:c+3] == 15):
                new_grid[r:r+3, c:c+3] = 5
                break # Only remove one
    
    # Re-apply the new block on top so it stays 15
    start_row, start_col = py - 1, px - 1
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if 0 <= r < 64 and 0 <= c < 64:
                new_grid[r, c] = 15

    # Handle the bottom row (r63) change: it seems to be a progress bar or counter.
    # The delta shows r63c62:5x2 -> r63c60:5x2 -> r63c58:5x2...
    # It moves left by 2 cells each time.
    # Let's find where the '5x2' is currently and move it.
    current_pos = -1
    for c in range(64):
        if grid[63, c] == 5 and (c == 0 or grid[63, c-1] != 5):
            # Found a start of a run of 5s. Check if length is 2.
            length = 0
            while c + length < 64 and grid[63, c+length] == 5:
                length += 1
            if length == 2:
                current_pos = c
                break
    
    if current_pos != -1:
        # Remove old 5s
        new_grid[63, current_pos:current_pos+2] = 0 # Or whatever background color was there
        # Add new 5s shifted left by 2
        new_pos = max(0, current_pos - 2)
        new_grid[63, new_pos:new_pos+2] = 5
    else:
        # Initial case for r63
        new_grid[63, 62:64] = 5

    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it's when a certain pattern is achieved.
    # Based on the game name 'su15', maybe all blocks are collected?
    # For now, return False as no win condition is obvious.
    return False