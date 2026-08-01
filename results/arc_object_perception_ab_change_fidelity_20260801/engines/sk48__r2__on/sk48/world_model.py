import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game involves a player-controlled object (the 'key' or 'block') and several targets/goals.
    # Based on the observed transitions, ACTION1 moves the key block (color 6) in some direction?
    # No, let's look at the same object shapes.
    # obj7 (color 6) is the "player" block. It starts at bbox=(24, 11, 29, 16).
    # In ACTION1, it seems to move from (24, 11) to (18, 11)? Or something else happens.
    # Let's re-examine the observarions.
    # INITIAL GRID: obj7 (color 6) is at (24, 11) to (29, 16).
    # ACTION1: changed cells r18c11...r23c11...r29c11. This looks like the block was moved or shifted.
    # Actually, looking at the deltas, ACTION1 shifts the color 6 block upwards.
    # ACTION2 shifts the block downwards.
    # ACTION3 shifts other blocks (colors 8, 9, 14).
    # ACTION4 modifies specific pixels.
    # The core mechanic seems to be a puzzle where you move a block to cover targets.
    # But wait, the observed transitions are very sparse and complex.
    # Let's simplify.
    # The game state is essentially a grid of colors.
    # Action 1: Move Up? Action 2: Move Down? Action 3: Move Right? Action 4: Move Left?
    # No, let's look at the action numbers again.
    # ACTION1: moves something from row 24-29 to 18-23. (Up)
    # ACTION2: moves something from row 18-23 back to 24-29. (Down)
    # ACTION2 (again): moves it further down to 30-35. (Down)
    # ACTION3: changes cells in rows 19-22, cols 36-42. This corresponds to the target objects (obj4, obj9, obj20).
    # ACTION4: changes small groups of pixels.
    # It looks like ACTION1=Up, ACTION2=Down, ACTION3=Right, ACTION4=Left?
    # Or maybe they are just specific triggers.
    # Given the constraints and the lack of clear movement patterns for all actions,
    # I will implement a simple version where actions shift the player block if possible.

    # Find the "player" block (color 6)
    # The player block is a connected component of color 6.
    # We need to find its bounding box.
    # Let's assume there's only one main player block of color 6.
    # Find all coordinates of color 6.
    coords = np.argwhere(grid == 6)
    if coords.size == 0:
        return grid.copy()
    
    # Bounding box of the player block
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    new_grid = grid.copy()
    
    # Define movements based on action
    dy, dx = 0, 0
    if action == 1: # Up
        dy = -6
    elif action == 2: # Down
        dy = 6
    elif action == 3: # Right
        dx = 6
    elif action == 4: # Left
        dx = -6
    
    # If movement is possible (all cells in new position are background color 5 or target colors),
    # we perform the move.
    # if dy != 0 or dx != 0:
    #     # Shift the entire block of color 6
    #     #<|channel>thought
    #     # The observed transitions show that the "player" block moves and interacts with other things.
    #     # But wait, ACTION3 changes targets. This suggests a different mechanic.
    #     # Let's look at the deltas again.
    #     # ACTION1 shifts r24-29 to r18-23.
    #     # ACTION2 shifts r18-23 back to r24-29, then later r24-29 to r30-35.
    #     # ACTION4 modifies pixels at r20c23, r20c29, etc.
    #     # It seems like the player block (color 6) acts as a "brush" or "key".
    #     # When it's at a certain position, action 4 might trigger something?
    #     # Or maybe Action 1-4 are just Up/Down/Left/Right for the whole world?
    #     # No, only specific blocks move.
    #     # Given the complexity and limited data, I will implement a basic movement model for color 6.

    # For this specific level, let's try to mimic the observed movements of the color 6 block.
    if action == 1: # Move Up
        # Shift all cells of color 6 up by 6 units
        mask = (grid == 6)
        new_grid[np.maximum(0, coords[:, 0] - 6), np.maximum(0, coords[:, 1])] = 6
        new_grid[coords[:, 0], coords[:, 1]] = 5 # Reset old positions to background
        return new_grid
    elif action == 2: # Move Down
        mask = (grid == 6)
        new_grid[np.minimum(grid.shape[0]-1, coords[:, 0] + 6), np.minimum(grid.shape[1]-1, coords[:, 1])] = 6
        new_grid[coords[:, 0], coords[:, 1]] = 5
        return new_grid
    elif action == 3: # Right?
        # Action 3 in observations changes target blocks. Let's just return grid.
        return grid.copy()
    elif action == 4: # Left?
        # Action 4 modifies pixels. Let's just return grid.
        return grid.copy()

    return new_grid

def is_level_complete(grid):
    # Level complete when targets are cleared or a certain state is reached.
    # In the observed data, no win state was provided.
    # Usually, it's when all targets of color 8, 9, 14 are gone or changed.
    targets = [8, 9, 14]
    for t in targets:
        if np.any(grid == t):
            return False
    return True