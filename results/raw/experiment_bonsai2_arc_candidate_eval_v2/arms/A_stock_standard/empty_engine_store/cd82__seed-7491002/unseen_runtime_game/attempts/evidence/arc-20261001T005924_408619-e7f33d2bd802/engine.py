import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    h, w = g.shape
    
    # Track a counter at bottom-right corner that decrements on each action
    # This appears to be a move counter or timer
    if action in [1, 2, 3, 4, 5]:
        # Directional actions: 1=up, 2=left, 3=right, 4=down, 5=up (based on observation)
        pass
    elif action == 6 and data is not None:
        # Click action - just decrement counter
        pass
    
    # The key pattern observed: every action causes r63c<col> to change from 4->5
    # where col starts at 63 and decreases by 1 each time
    # Let's find the current counter value by looking at row 63
    # Row 63 initially is all 4s. As moves happen, cells turn to 5 from right to left.
    
    # Find the rightmost cell in row 63 that is still 4 (or first 5 from right)
    # Actually, let's count how many 5s are in row 63 starting from the right
    # Initially row 63 is all 4s (64 cells of color 4)
    
    # Count consecutive 5s from the right end of row 63
    count_5_from_right = 0
    for c in range(w-1, -1, -1):
        if g[63][c] == 5:
            count_5_from_right += 1
        else:
            break
    
    # If there's a next cell to flip (the one just to the left of the 5-block)
    if count_5_from_right < w:
        target_col = w - 1 - count_5_from_right
        if g[63][target_col] == 4:
            g[63][target_col] = 5
    
    return g

def is_level_complete(grid):
    # Win state: check if the entire bottom-right area has been filled with 5s
    # Based on observations, the game seems to complete when certain conditions are met
    # The simplest general rule: level is complete when row 63 is entirely 5s
    h, w = grid.shape
    if h > 0 and w > 0:
        if np.all(grid[h-1, :] == 5):
            return True
    return False