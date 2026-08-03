import numpy as np

def engine(grid, action, data):
    """
    World model for game 'tu93'.
    The grid contains a maze of colors. The player controls an entity (represented by color 9).
    Based on the observed transitions:
    ACTION2: Down
    ACTION3: Left
    ACTION4: Right
    The entity moves between designated slots (3x3 blocks).
    Additionally, there's a counter/progress bar at r63 that decrements or changes.
    """
    new_grid = grid.copy()
    
    # Find current position of the agent (color 9)
    # Since it's usually in 3x3 blocks, we find the top-left corner of the block
    coords = np.argwhere(grid == 9)
    if coords.size == 0:
        return new_grid
    
    # Agent is represented by a 3x3 block of color 9 (mostly)
    # We identify the center or top-left to determine movement
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    
    # Current slot boundaries
    curr_r = r_min
    curr_c = c_min

    # Movement logic based on observations
    # ACTION2 -> Move Down
    # ACTION3 -> Move Left
    # ACTION4 -> Move Right
    dr, dc = 0, 0
    if action == 2:
        dr = 12 # Observed jump from r21->r27, r27->r33, r33->r39, r39->r45
    elif action == 3:
        dc = -6 # Observed jump from c33->c27, c27->c21
    elif action == 4:
        dc = 6  # Observed jump from c21->c27, c27->c33
    else:
        return new_grid

    new_r, new_c = curr_r + dr, curr_c + dc
    
    # Bounds check (simple)
    if not (0 <= new_r < 64 and 0 <= new_c < 64):
        return new_grid

    # Clear old position (set to background color 5 or the slot's original structure)
    # In this specific game, slots are often replaced by 0x3 or 5x3.
    # To be safe, we look at what was there before the agent arrived if possible, 
    # but based on deltas, it seems they toggle between 9 and 0/5.
    for r in range(curr_r, curr_r + 3):
        for c in range(curr_c, curr_c + 3):
            if 0 <= r < 64 and 0 <= c < 64:
                # The delta shows that when moving away, cells become 0 or 5.
                # We use a heuristic: if it's a path, it becomes 0; otherwise 5.
                new_grid[r, c] = 0 if grid[r, c] == 9 else grid[r, c]

    # Set new position to color 9
    for r in range(new_r, new_r + 3):
        for c in range(new_c, new_c + 3):
            if 0 <= r < 64 and 0 <= c < 64:
                new_grid[r, c] = 9

    # Handle the progress bar at r63 (observed as changing from 6x56,0x8 -> ...0x1 etc)
    # It seems to be a countdown of remaining steps or distance.
    # Every action reduces the count of '6's at the end of row 63.
    row63 = new_grid[63].copy()
    sixes = np.where(row63 == 6)[0]
    if sixes.size > 0:
        last_six = sixes[-1]
        row63[last_six] = 0
        new_grid[63] = row63

    return new_grid

def is_level_complete(grid):
    """
    The level is complete when the agent reaches a target state.
    Based on observations, we don't have a win state grid, but usually it involves
    reaching a specific coordinate or clearing the progress bar.
    """
    # Heuristic: if color 9 is in the bottom-most slot area and progress bar is empty.
    coords = np.argwhere(grid == 9)
    if coords.size == 0:
        return False
    r_min = coords.min(axis=0)[0]
    # Target slots are around r45
    if r_min >= 45:
        # Check if progress bar (r63) has no more 6s
        if not np.any(grid[63] == 6):
            return True
    return False