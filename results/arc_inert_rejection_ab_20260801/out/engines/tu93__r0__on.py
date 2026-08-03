import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, it seems like a puzzle where an object (represented by color 9/4)
    # moves across different 'slots' or 'paths'.
    # The state of the same board is likely tracked in some other way, but based on the
    # delta changes, wes see that a specific pattern (color 9/4) moves.
    # Action 2 (Down), 3 (Left), 4 (Right), 1 (Up) are directional movements.
    # Color 9/4 represents the player/object moving between slots.
    # 
    # Let's identify the current position of the "player" (the 3x3 block of color 9/4).
    # In the initial grid, there is a 3x3 block of color 9 at (21, 33) to (23, 35).
    # Initial positions of interest:
    # r21-23 c33-35: value 9
    # r27-29 c33-35: value 5 (which will be changed to 9)
    # r33-35 c33-35: value 5 (which will be}
    # r39-41 c33-35: value 5 (<|channel>thought
    # r45-47 c33-35: value 5 (// this seems like be a target slot)

    # Find the center of the 3x3 block of color 9 or 4.
    # Find all cells with values 9 or 4.
    # Find the most common row and column indices.
    # find the center coordinate (y, x)
    rows, cols = np.where((grid == 9) | (grid == 4))
    if len(rows) == 0:
        return grid
    
    cy, cx = int(np.median(rows)), int(np.median(cols))
    
    # Define movement vectors for actions
    # Action 1: Up, 2: Down, 3: Left, 4: Right
    moves = {
        1: (-6, 0), # Moves by 6 units in y
        2: (6, 0),  # Moves by 6 units in y
        3: (0, -6), # Moves by 6 units in x
        4: (0, 6),  # Moves by 6 units in x
    }
    
    if action not in moves:
        return grid
    
    dy, dx = moves[action]
    ny, nx = cy + dy, cx + dx
    
    # The "player" is a 3x3 block. We need to clear the old position and restore it to color 5.
    # Let's assume the target slot must be valid (i.e., not blocked).
    # Note: In the observed transitions, we see that the player block changes slightly (9x3, then 4x1, etc.)
    # but generally stays as a 3x3 block of color 9/4.
    # We also observe r63c... changing values from 5 to 0. This looks like a progress bar or same-state indicator.
    # Win state is likely when the object reaches a specific location.
    
    # Create a copy of the grid
    new_grid = grid.copy()
    
    # Clear current position (set back to background color 5)
    # Based on deltas, cells are set to 0 if they were part of the path? No, let's look closer.
    # Initial: r21-23 c33-35 was 9. Action 2 -> r21-23 c33-35 becomes 0.
    # Then r27-29 c33-35 becomes 9.
    # So it seems old positions become 0 and new positions become 9.
    
    # Find all coordinates of the "player" block
    player_coords = np.where((grid == 9) | (grid == 4))[0] # this is wrong logic
    # Let's just use the median center and clear a 3x3 area around it.
    for y in range(cy - 1, cy + 2):
        for x in range(cx - 1, cx + 2):
            if 0 <= y < 64 and 0 <= x < 64:
                new_grid[y, x] = 0
                
    # Set new position to 9
    # The observed transitions show some variation (e.g., 4x1), but we'll stick to 3x3 color 9.
    for y in range(ny - 1, ny + 2):
        for x in range(nx - 1, nx + 2):
            if 0 <= y < 64 and 0 <= x < 64:
                new_grid[y, x] = 9
    
    # Update progress bar at r63
    # In deltas: r63c55:0x1 -> r63c54:0x1 -> r63c52:0x2 ...
    # It seems every move decrements the count of cells with value 5 or increments zeros from right to left.
    # Let's find current zero boundary on row 63.
    zero_cols = np.where(grid[63] == 0)[0]
    if len(zero_cols) > 0:
        max_z = np.max(zero_cols)
    else:
        max_z = 55 # starting point based on initial grid r63:6x56,0x8 (so cols 56-63 are 0)
    
    # Every action reduces the "filled" part of row 63 by some amount.
    # Action 2: r63c55:0x1 (one cell becomes 0)
    # Action 3: r63c52:0x2 (two cells become 0)
    # This is a bit inconsistent but let's just decrement the first '5' from the right.
    for x in range(63, -1, -1):
        if new_grid[63, x] == 5:
            new_grid[63, x] = 0
            break

    return new_grid

def is_level_complete(grid):
    # Win state is likely when the player reaches the target slot (r45-47 c33-35).
    rows, cols = np.where((grid == 9) | (grid == 4))
    if len(rows) == 0:
        return False
    cy, cx = int(np.median(rows)), int(np.median(cols))
    # Target center: y=46, x=34
    return cy == 46 and cx == 34