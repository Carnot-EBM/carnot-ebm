import numpy as np

def engine(grid, action, data):
    """
    The observed transitions show that clicking on specific regions of the grid 
    causes changes in other distant parts of the grid and moves a marker at the bottom row.
    Specifically, Action 6 (click) triggers these updates.
    Looking at the deltas:
    - Clicking around x=48, y=21 affects rows 9, 10, 11 (a structure near the top).
    - Clicking around x=24, y=47 affects rows 34-41 (a structure in the middle).
    - Every click also shifts a value '4' in the last row (r63) to the left.
    
    This suggests a puzzle where interacting with certain objects modifies their state 
    and progresses a counter/marker. Since we are inducing an executable world model 
    from limited samples, we implement the logic based on the provided delta patterns.
    """
    if action != 6:
        return grid.copy()

    new_grid = grid.copy()
    px, py = data['x'], data['y']

    # Marker movement in r63
    # The marker is color 4. It moves left by some amount each time.
    # We find the current position of color 4 in the last row.
    last_row = new_grid[-1]
    marker_pos = np.where(last_row == 4)[0]
    if len(marker_pos) > 0:
        curr_col = marker_pos[0]
        # Based on deltas, it moves left. Let's determine how much.
        # In the first few clicks, it moved from c63 -> c61 (2), then c60 (1), c59 (1)...
        # This looks like a sequence of shifts. To be general, we shift it left.
        # For this specific game state transition, let's simulate a move of 1 or 2.
        shift = 1 if curr_col < 60 else 2 # Heuristic based on observed delta jumps
        new_col = max(0, curr_col - shift)
        new_grid[-1, curr_col] = 3 # Reset old to background
        new_grid[-1, new_col] = 4

    # Effect on grid structures
    # If clicking top-right area (x=48, y=21)
    if px >= 40 and py <= 30:
        # The deltas show color 14 filling in rows 9, 10, 11 moving rightwards.
        # We find where the 'gap' is and fill it.
        for r in [9, 10, 11]:
            row = new_grid[r]
            # Find first occurrence of something that isn't 14 but should be part of the object
            # In these rows, colors are 5 (bg), 3, 14, 13.
            # We look for indices where we can place 14s.
            cols = np.where((row == 5) & (np.arange(64) > 27))[0]
            if len(cols) > 0:
                target = cols[0]
                # Fill a small block as seen in deltas
                new_grid[r, target:min(target+3, 64)] = 14

    # If clicking middle-left area (x=24, y=47)
    elif px <= 30 and py >= 40:
        # Affects rows 34-41. Color 11 fills in.
        for r in range(34, 42):
            row = new_grid[r]
            cols = np.where((row == 5) & (np.arange(64) > 9))[0]
            if len(cols) > 0:
                target = cols[0]
                new_grid[r, target:min(target+3, 64)] = 11

    return new_grid

def is_level_complete(grid):
    """
    The level is likely complete when the marker at the bottom reaches a certain point
    or all structures are filled. Based on typical ARC patterns, we check if the 
    marker has reached the far left or a specific goal.
    """
    last_row = grid[-1]
    marker_pos = np.where(last_row == 4)[0]
    if len(marker_pos) > 0 and marker_pos[0] <= 0:
        return True
    return False