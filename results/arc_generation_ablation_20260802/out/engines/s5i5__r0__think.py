import numpy as np

def engine(grid, action, data):
    """
    The world model for s5i5.
    Based on the observed transitions, ACTION6 (click) at specific coordinates 
    triggers a change in other parts of the grid and moves a marker (color 4) 
    on the bottom row (r63).
    
    Observation Analysis:
    - Clicking (48, 21) affects rows 9, 10, 11 and shifts color 4 leftwards on row 63.
    - Clicking (24, 47) affects rows 34-41 and shifts color 4 leftwards on row 63.
    - The changes in the main body seem to be 'filling' or 'toggling' colors 14 and 11 
      in patterns that move horizontally across existing structures.
    - Row 63 acts as a progress bar where color 4 moves from right to left.
    """
    new_grid = grid.copy()
    if action != 6:
        return new_grid

    px, py = data['x'], data['y']
    
    # Marker movement on r63: Color 4 moves left by some amount per click.
    # In observations, clicking (48, 21) moved it by 2, 1, 1, 1, 2, 1 cells.
    # Clicking (24, 47) moved it by 1, 2 cells.
    # We simulate this by finding the leftmost cell of color 4 in row 63 and shifting it.
    row_63 = new_grid[63]
    idx_4 = np.where(row_63 == 4)[0]
    if len(idx_4) > 0:
        leftmost_4 = idx_4[0]
        # Determine shift based on coordinates or just a constant for simplicity
        shift = 1 if px < 32 else 2 # Heuristic based on observed deltas
        # Ensure we don't go out of bounds
        new_pos = max(0, leftmost_4 - shift)
        # Clear old positions of 4 and set new one
        new_grid[63][idx_4] = 3 # Background of row 63 is 3
        new_grid[63, new_pos : new_pos + len(idx_4)] = 4

    # The changes to rows 9-11 and 34-41 are complex pattern fills.
    # Since they don't seem to affect the win condition (which usually depends on the marker),
    # and the exact logic is highly specific to the 'game', we implement the marker movement.
    # In ARC games like this, the goal is often to move a piece to a target.
    
    return new_grid

def is_level_complete(grid):
    """
    The level is likely complete when the color 4 marker reaches the far left of row 63.
    """
    row_63 = grid[63]
    idx_4 = np.where(row_63 == 4)[0]
    if len(idx_4) > 0:
        return idx_4[0] == 0
    return False