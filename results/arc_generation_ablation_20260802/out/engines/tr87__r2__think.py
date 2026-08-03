import numpy as np

def engine(grid, action, data):
    """
    World model for ARC game tr87.
    Based on the observed transitions:
    ACTION4 seems to move a set of blocks horizontally in specific rows (r48, r49, r59, r60).
    ACTION1 and ACTION2 seem to modify colors/patterns in the lower grid area (r52-r56).
    The cell at (63, 63) changes value based on actions, acting like a counter or state tracker.
    """
    new_grid = grid.copy()
    h, w = new_grid.shape

    if action == 4:
        # Action 4 moves a pattern of color 0s and 3s across columns.
        # Looking at deltas: c15->c22->c29... it's moving by 7 units.
        # The delta shows r48c15:3x5, r48c22:0x5 etc.
        # This suggests shifting a block of width 5 from col x to x+7.
        # We need to identify where the 'active' block is.
        # In this specific sequence, it starts at 15, then 22, then 29.
        
        # Find current position of the block in row 48 (color 3)
        current_col = -1
        for c in range(w):
            if new_grid[48, c] == 3:
                current_col = c
                break
        
        if current_col != -1:
            # Move block of 5 cells forward by 7
            target_col = current_col + 7
            if target_col + 5 <= w:
                # Clear old positions
                new_grid[48, current_col : current_col + 5] = 2 # background color for these rows seems to be 3 or 2? No, looking at INITIAL grid, r48 is mostly 3s.
                # Wait, INITIAL grid says r48: 3x15, 0x5, 3x44. So base is 3, block is 0.
                # Delta ACTION4: r48c15:3x5 r48c22:0x5. This means it fills the gap with 3 and creates a new gap of 0.
                
                # Correct logic based on deltas:
                # Row 48 & 60: Block of 5 cells moves from x to x+7.
                # Row 49 & 59: Single cells move from x to x+7 (approx).
                
                # Let's apply the specific delta pattern observed
                # Shift row 48/60 blocks
                new_grid[48, current_col : current_col + 5] = 3
                new_grid[60, current_col : current_col + 5] = 3
                if target_col + 5 <= w:
                    new_grid[48, target_col : target_col + 5] = 0
                    new_grid[60, target_col : target_col + 5] = 0
                
                # Shift row 49/59 points
                # Looking at r49c15:3x1 r49c19:3x1 r49c22:0x1 r49c26:0x1
                # It seems it clears two points and sets two new ones.
                for r in [49, 59]:
                    # Find indices of color 0 in these rows
                    zeros = np.where(new_grid[r] == 0)[0]
                    for z in zeros:
                        if z + 7 < w:
                            new_grid[r, z] = 3
                            new_grid[r, z+7] = 0

    elif action == 2:
        # Action 2 modifies the pattern in r52-r56.
        # Based on deltas, it changes colors between 5 and 7.
        # Also decrements (or increments) the value at (63, 63).
        # Since we don't have a clear rule for the pattern change, 
        # we simulate the state tracker if possible.
        if new_grid[63, 63] > 0:
            new_grid[63, 63] -= 1
        else:
            new_grid[63, 63] = 4 # wrap around or reset

    elif action == 1:
        # Action 1 also modifies patterns in r52-r56.
        if new_grid[63, 63] > 0:
            new_grid[63, 63] -= 1
        else:
            new_grid[63, 63] = 4

    return new_grid

def is_level_complete(grid):
    # No win state provided in observed transitions.
    # Usually ARC levels are complete when a specific target pattern is reached.
    # Without a WIN STATE grid, we return False unless a common condition is met.
    return False