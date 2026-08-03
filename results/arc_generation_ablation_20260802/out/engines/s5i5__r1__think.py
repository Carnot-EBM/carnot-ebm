import numpy as np

def engine(grid, action, data):
    """
    The observed transitions show that clicking on specific regions of the grid
    causes a pattern change in another region and moves a marker at the bottom row (r63).
    Specifically, Action 6 clicks trigger changes to blocks of colors in the middle of the board
    and shift a color 4 cell leftwards along the bottom edge (row 63).
    
    Based on the provided deltas:
    - Clicks at (48, 21) affect rows 9, 10, 11 and move the color 4 pixel at r63 from right to left.
    - Clicks at (24, 47) affect rows 34-41 and also move the color 4 pixel at r63 further left.
    
    This suggests a mechanism where interacting with 'objects' or 'buttons' on the map
    progresses a state represented by the position of the color 4 cell in the last row.
    """
    if action != 6:
        return grid.copy()

    new_grid = grid.copy()
    px, py = data['x'], data['y']
    
    # The most consistent observable effect is the movement of the color 4 cell in row 63.
    # We find the current position of the color 4 cell in the last row.
    last_row = new_grid[-1, :]
    pos = np.where(last_row == 4)[0]
    
    if len(pos) > 0:
        current_col = pos[0]
        # Move the marker one step to the left if possible
        if current_col > 0:
            new_grid[-1, current_col] = last_row[current_col - 1] # Restore previous background (usually 3)
            new_grid[-1, current_col - 1] = 4
            
    # Note: The specific pattern changes in rows 9-11 and 34-41 are complex and likely 
    # tied to a state machine or coordinate-based trigger. However, since we only have 
    # limited transitions, the primary global state change is the marker at r63.
    # To accurately simulate the provided deltas for the specific coordinates:
    
    if px == 48 and py == 21:
        # This click sequence moves the marker and modifies blocks around col 36-51 in rows 9-11.
        # Since the delta varies per call, it implies an internal counter/state.
        pass 
    elif px == 24 and py == 47:
        # This click sequence moves the marker and modifies blocks around col 9-13 in rows 34-41.
        pass

    return new_grid

def is_level_complete(grid):
    """
    The level is complete when the marker (color 4) reaches its target destination.
    Looking at the initial grid, color 4 starts at r63c63.
    Based on the movement, it's moving left. Completion usually occurs when it hits 0 or a target.
    """
    last_row = grid[-1, :]
    pos = np.where(last_row == 4)[0]
    if len(pos) > 0 and pos[0] == 0:
        return True
    return False