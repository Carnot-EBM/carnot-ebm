import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # The observed transitions show that clicking on specific areas triggers a change in the 
    # layout of colors and moves a marker at the bottom row (r63).
    # The target coordinates provided in thes observations are (48, 21) and (24, 47).
    # These correspond to "regions" or "buttons" in the game world.
    # Let's analyze the same coordinate clicks.
    # Each click on (48, 21) shifts some patterns of color 14 in rows 9-11.
    #<|channel>thought
    # Based on the delta descriptions, ACTION6 (click) modifies cells in specific regions.
    # We can see that it changes values to 14 in rows 9-11 and 11 in rows 34-45.
    # And it always updates r63 (the progress bar/marker).
    # Since we don't have enough information to deduce a complex logic for the movement of the 
    # pattern shift, we will implement a simple state-based transition based on the observed deltas.
    
    new_grid = grid.copy()
    x, y = data['x'], data['y']
    
    if x == 48 and y == 21:
        # This is a region that affects rows 9-11 and moves marker at r63.
        # The marker at r63 seems to be moving leftwards from col 63.
        # Find current position of color 4 in row 63.
        marker_pos = np.where(grid[63] == 4)[0]
        if len(marker_pos) > 0:
            marker_pos = marker_pos[0]
            # Move marker left by some amount (usually 1 or 2)
            # Shift patterns in rows 9-11
            # We can observe that the first click shifted cells to c36, then c39, etc.
            #<|channel>thought
            # Let's refine this. Each click on (48, 21) shifts a "block" of color 14.
            # In the observations, it starts at c36, then c39, c42, c45, c48, c51.
            # It looks like it moves by 3 columns each time.
            # Current block start column for color 14 in row 9 is found by looking for the first '14'.
            # But wait, the initial grid has 14s already.
            # The deltas show new 14s appearing and replacing others.
            # Let's just simulate the movement of the marker in r63 as the primary state.
            
            new_grid[63, :] = 3
            # Find how many 4s were there and move them.
            num_fours = np.sum(grid[63] == 4)
            # Move current 4s one step left or similar.
            # Based on observed: r63c61:4x2 -> r63c60:4x1 -> r63c59:4x1...
            # This suggests the total number of 4s might be decreasing or shifting.
            # Actually, let's look at the indices: 61 (count 2), 60 (count 1), 59 (count 1)...
            # It seems to be a single cursor moving left.
            cursor = np.where(grid[63] == 4)[0]
            if len(cursor) > 0:
                curr_col = cursor[-1]
                new_col = curr_col - 1 if curr_col > 0 else 0
                new_grid[63, new_col] = 4
        return new_grid

    if x == 24 and y == 47:
        # Similar logic for the other button.
        cursor = np.where(grid[63] == 4)[0]
        if len(cursor) > 0:
            curr_col = cursor[-1]
            new_col = curr_col - 1 if curr_col > 0 else 0
            new_grid[63, new_col] = 4
        return new_grid
        
    return new_grid

def is_level_complete(grid):
    # The level is likely complete when the marker in r63 reaches the far left.
    marker_pos = np.where(grid[63] == 4)[0]
    if len(marker_pos) > 0:
        return marker_pos[0] == 0
    return False