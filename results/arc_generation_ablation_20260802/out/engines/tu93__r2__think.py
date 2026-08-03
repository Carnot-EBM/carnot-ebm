import numpy as np

def engine(grid, action, data):
    """
    The game appears to be a puzzle where ACTION2 (Down), ACTION3 (Left), 
    ACTION4 (Right) move a 'block' or 'cursor' of color 9 across a grid.
    Looking at the deltas:
    - ACTION2 moves the 3x3 block of 9s down by some distance.
    - ACTION3 moves it left.
    - ACTION4 moves it right.
    - The cells they leave behind are reset to 0 (or their background).
    - There is also a counter/progress bar at r63c50+ that decrements (color changes to 0).
    """
    new_grid = grid.copy()
    h, w = new_grid.shape
    
    # Find current position of the 9s block
    coords = np.argwhere(grid == 9)
    if coords.size == 0:
        return new_grid
    
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    
    # Define movement offsets based on observed transitions
    # Action 2: Down
    # Action 3: Left
    # Action 4: Right
    dr, dc = 0, 0
    if action == 2:
        dr = 6 # Observed jumps from r21->r27, r27->r33, r33->r39, r39->r45
    elif action == 3:
        dc = -6 # Observed jumps from c33->c27, c27->c21
    elif action == 4:
        dc = 6 # Observed jumps from c21->c27, c27->c33

    # Clear old block (set to 0 as seen in deltas)
    new_grid[r_min:r_max+1, c_min:c_max+1] = 0
    
    # Calculate new position
    nr_min, nc_min = r_min + dr, c_min + dc
    nr_max, nc_max = r_max + dr, c_max + dc
    
    # Clip to grid boundaries
    nr_min = max(0, min(h-1, nr_min))
    nc_min = max(0, min(w-1, nc_min))
    nr_max = max(0, min(h-1, nr_max))
    nc_max = max(0, min(w-1, nc_max))
    
    # Place new block of 9s
    new_grid[nr_min:nr_max+1, nc_min:nc_max+1] = 9
    
    # Handle the progress bar at row 63
    # The observed transitions show cells at r63 changing from some value to 0.
    # This happens every time an action is taken.
    # We look for the rightmost non-zero cell in row 63 and clear it.
    row_63 = new_grid[63, :]
    nonzero_indices = np.where(row_63 != 0)[0]
    if nonzero_indices.size > 0:
        last_idx = nonzero_indices[-1]
        # Based on deltas like "r63c55:0x1", "r63c54:0x1", etc.
        # It clears one or more cells moving leftwards.
        # To match exactly, we can just set a few pixels to 0 based on movement.
        # But since we don't have the exact logic for the bar, clearing the last 
        # active pixel is a reasonable approximation of 'progress'.
        new_grid[63, last_idx] = 0

    return new_grid

def is_level_complete(grid):
    """
    The level is likely complete when the progress bar (row 63) is fully cleared
    or the block reaches a specific target. Given the data, we check if the 
    bar at r63 is empty.
    """
    return np.all(grid[63, 48:] == 0)