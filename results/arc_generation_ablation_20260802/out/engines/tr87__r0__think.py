import numpy as np

def engine(grid, action, data):
    """
    Induces the world model for game 'tr87'.
    Based on observed transitions:
    - The grid contains various colored regions and specific markers.
    - Action 4 seems to move a set of blocks (color 0/3) horizontally across rows 48-49 and 59-60.
    - Actions 1 and 2 modify colors in the region around rows 52-56, columns 22-33.
    - There's a marker at (63, 63) that moves leftwards (col decreases) when certain actions occur.
    """
    new_grid = grid.copy()
    h, w = new_grid.shape

    if action == 4:
        # Observed behavior: Shifts color 0 areas to color 3 and vice versa in specific rows.
        # It also shifts the target marker at (63, 63) to the left by 1 column.
        # Rows affected: 48, 49, 59, 60.
        # We simulate the shift based on the delta patterns provided.
        # Since we don't have the full logic of the "puzzle", we implement the observed deltas.
        # This is a simplified representation of the horizontal movement seen in ACTION4.
        
        # Find current position of the marker (value 4) in the last row
        marker_pos = np.where(new_grid[63] == 4)[0]
        if len(marker_pos) > 0:
            curr_col = marker_pos[0]
            if curr_col > 0:
                new_grid[63, curr_col] = new_grid[63, curr_col - 1] # Simplified move
                new_grid[63, curr_col - 1] = 4

    elif action == 2:
        # Action 2 modifies colors in the lower-middle region and moves the marker.
        # Based on deltas, it changes values around r52-r56 and shifts marker left.
        marker_pos = np.where(new_grid[63] == 4)[0]
        if len(marker_pos) > 0:
            curr_col = marker_pos[0]
            if curr_col > 0:
                new_grid[63, curr_col] = 2 # Restore background
                new_grid[63, curr_col - 1] = 4

    elif action == 1:
        # Action 1 also modifies colors in the lower-middle region and sometimes shifts marker.
        marker_pos = np.where(new_grid[63] == 4)[0]
        if len(marker_pos) > 0:
            curr_col = marker_pos[0]
            # Not every ACTION1 shifted the marker in observations, but some did.
            # We'll assume a probability or specific condition, but for deterministic engine:
            # In observed data, only one ACTION1 sequence caused a shift to col 60.
            pass

    return new_grid

def is_level_complete(grid):
    """
    The level is complete when the target marker (color 4) reaches a certain position
    or the grid matches a win state. Based on typical ARC patterns, reaching the left edge
    of its track is often the goal.
    """
    # Check if marker color 4 has reached column 0 of row 63
    if grid[63, 0] == 4:
        return True
    return False