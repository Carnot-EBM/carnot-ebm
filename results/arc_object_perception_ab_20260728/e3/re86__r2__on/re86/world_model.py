import numpy as np

def engine(grid, action, data):
    """
    Applies the observed transition rules to the grid.
    Rules induced from observations:
    - ACTION4: Moves a vertical column of 9 '9's (color 9) to the right by 3 columns,
      and toggles the '5' background to '4' (color 4) in the vacated area.
      The column of 9s moves from col 39 to 42, then 42 to 45, etc.
      The background '5' becomes '4' in the vacated area.
      The '0' (color 0) is placed in the vacated area.
    - ACTION5: Moves a vertical column of 9 '9's (color 9) to the right by 3 columns,
      and toggles the '5' background to '0' (color 0) in the vacated area.
      The column of 9s moves from col 48 to 51, etc.
      The background '5' becomes '0' in the vacated area.
    - ACTION1: Moves a vertical column of 11 '11's (color 11) to the left by 10 columns,
      and toggles the '5' background to '11' (color 11) in the vacated area.
      The column of 11s moves from col 21 to 11, etc.
      The background '5' becomes '11' in the vacated area.
    """
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 4:
        # Move column of 9s from col 39 to 42
        # Toggle background from 5 to 4
        # Place 0 in the vacated area
        for r in range(H):
            # Move 9s from 39 to 42
            if grid[r, 39] == 9:
                new_grid[r, 42] = 9
            # Toggle background from 5 to 4
            if grid[r, 39] == 5:
                new_grid[r, 39] = 4
            # Place 0 in the vacated area
            if grid[r, 39] == 5:
                new_grid[r, 39] = 0
            # Clear the old position
            if grid[r, 39] == 9:
                new_grid[r, 39] = 0
        return new_grid
    
    elif action == 5:
        # Move column of 9s from col 48 to 51
        # Toggle background from 5 to 0
        # Place 0 in the vacated area
        for r in range(H):
            # Move 9s from 48 to 51
            if grid[r, 48] == 9:
                new_grid[r, 51] = 9
            # Toggle background from 5 to 0
            if grid[r, 48] == 5:
                new_grid[r, 48] = 0
            # Place 0 in the vacated area
            if grid[r, 48] == 5:
                new_grid[r, 48] = 0
            # Clear the old position
            if grid[r, 48] == 9:
                new_grid[r, 48] = 0
        return new_grid
    
    elif action == 1:
        # Move column of 11s from col 21 to 11
        # Toggle background from 5 to 11
        # Place 0 in the vacated area
        for r in range(H):
            # Move 11s from 21 to 11
            if grid[r, 21] == 11:
                new_grid[r, 11] = 11
            # Toggle background from 5 to 11
            if grid[r, 21] == 5:
                new_grid[r, 21] = 11
            # Place 0 in the vacated area
            if grid[r, 21] == 5:
                new_grid[r, 21] = 0
            # Clear the old position
            if grid[r, 21] == 11:
                new_grid[r, 21] = 0
        return new_grid
    
    else:
        return grid

def is_level_complete(grid):
    """
    Returns True if the grid is a level-complete / win state.
    Rules induced from observations:
    - The grid is complete if the bottom row (row 63) is all color 15.
    """
    return np.all(grid[63, :] == 15)