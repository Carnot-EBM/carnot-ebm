import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Inducing rules from observed transitions:
    # Action 4: Down/Right movement or a specific pattern change.
    # Action 2: Left/Up movement or a specific pattern change.
    # The game seems to be a puzzle where objects move and colors shift.
    # Based on the transition deltas, it's more than simple translation.
    #
    # However, since we can only induce general rules and the same state layout is used,
    # we can actually implement a logic that mimics the behavior of the same object moving.
    #
    # Let's look at the ACTION4 delta:
    # It changes cells in rows 21-37 and column 37-43.
    # It replaces some values with 15 (color 15) and others with 2 (color 2) or 5 (color 5).
    # It moves an "object" consisting of color 2 and 15.
    #
    # Looking at ACTION2:
    # It restores some cells to 5 (background) and 
    # moves the "object" further down and left.
    #
    # This looks like a player character (represented by color 2 or similar)
    # and its interaction with the same grid.
    #
    # Given the constraints and the<|channel>thought process, I will implement a basic movement system
    # for the "active" entity (the block of colors 2/15).
    
    new_grid = grid.copy()
    
    if action == 6: # Click
        px, py = data['x'], data['y']
        # Simple click logic: if clicking on something not background (5), it might change.
        # But we've only seen actions 2 and 4.
        pass

    # The observed transitions are highly specific. In a real ARC-AGI game, 
    # these would be move commands. Action 4 is likely 'Right', Action 2 is 'Left'.
    # Let's assume standard directional mapping: 2=Up, 3=Down, 4=Right, 1=Left?
    # Or maybe 2=Left, 4=Right.
    # Based on ACTION4 delta: r21c39 -> r37c43. It moves things to higher row/col indices.
    # Action 4: Right/Down. Action 2: Left/Up.
    
    # To properly simulate this without full physics, I will implement a simple shift
    # for the "active" object (non-background cells that aren't static walls).
    # Walls are colors 0, 3, 4. Background is 5.
    # Active objects are color 2, 15.
    
    # Find all active cells
    active_cells = np.argwhere(np.isin(grid, [2, 15]))
    if len(active_cells) == 0:
        return new_grid

    # Define movement vectors based on action
    # Action 4: Down-Right
    # Action 2: Up-Left (or just different direction)
    move_vec = np.array([0, 0])
    if action == 4:
        move_vec = np.array([1, 1])
    elif action == 2:
        move_vec = np.array([1, -1]) # Based on ACTION2 delta moving rows down and cols left
    
    # Apply movement to active cells
    for r, c in active_cells:
        nr, nc = r + move_vec[0], c + move_vec[1]
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            # Simple collision check with walls (0, 3, 4)
            if not np.isin(grid[nr, nc], [0, 3, 4]):
                new_grid[r, c] = 5
                new_grid[nr, nc] = grid[r, c]
    
    return new_grid

def is_level_complete(grid):
    # Level complete if the "active" object reaches a certain target or color changes.
    # In many ARC games, it's about clearing objects or reaching a goal.
    # Since no win state was provided, I will assume completion when the active object
    # moves into a specific region or all of a certain color are gone.
    # For now, return False as we don't have the win condition.
    return False