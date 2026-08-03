import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action == 4: # ACTION4 seems to be a rightward shift of some elements
        new_grid = grid.copy()
        # Identify the "active" region based on observed transitions
        # The changes occur in columns 11-47 and rows 14-31
        # Each ACTION4 shifts everything in that range by 1 column to the right
        # Shift logic for specific colors/objects
        # Note: The delta shows cells changing from color 0 or other values to 10, 
        # and others changing back to 0. This looks like a movement of a "gap" or "object".
        # Let's try a simple horizontal shift of a slice of the same width.
        # Shift a window of size 3x18 (approx) across the center.
        # Shift the entire content of the center area to the right by 1.
        # Shift theC center area boundaries are roughly r14-r31 and c11-c50.
        #
        # Based on the<|channel>thought process, it's more likely a player character moving right.
        # Find the object (color 0) and move it.
        # Move all pixels of color 0 within the central corridor.
        # Move all pixels of color 10 (walls) if they are part of the "moving block".
        # Looking at the deltas, the changes are very structured.
        # It's a shift of a vertical strip of width 3.
        #
        # Correct interpretation: ACTION4 is 'Right'.
        # There is a "player" object consisting of multiple cells.
        # The player moves one unit to the right.
        # Any cell that was part of the player moves to the new position.
        # The old position becomes the background/wall.
        # Let's implement a general shift for the active region.
        
        # Define the active window based on observed transitions
        # Rows 14 to 31 inclusive.
        # Columns 11 to 63.
        # Shift everything in this window to the right by 1.
        # Values shifted from column j to j+1.
        # New values for column 11 will be filled with the wall color (10).
        #
        # Based on the delta, it looks like a specific pattern is moving.
        #
        # Let's try shifting the entire central corridor content.
        # This matches the r14c11:10x3 etc patterns.
        
        window_rows = slice(14, 32)
        window_cols = slice(11, 64)
        
        # Save current state of the window
        window = grid[window_rows, window_cols].copy()
        
        # Shift right by 1
        shifted_window = np.full(window.shape, 10) # Fill with wall color
        shifted_window[:, 1:] = window[:, :-1]
        
        new_grid[window_rows, window_cols] = shifted_window
        return new_grid

    return grid

def is_level_complete(grid):
    # return True if `grid` is a level-complete / win state, else False.
    # No win state provided in observed transitions, but typically it involves 
    # reaching a goal or clearing objects.
    # For now, return False as no target was seen.
    return False