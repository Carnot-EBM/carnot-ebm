import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state based on the action and observed transitions.
    The game appears to be a puzzle where certain blocks of colors are moved or 
    toggled up/down/left/right.
    """
    # The board size is 64x64.
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1: # Move Up (or similar)
        # Based on ACTION1 delta: it changes cells in rows 18-29, cols 11-17.
        # It seems to actually shift a specific pattern of colors (the 'player' or 'object')
        # shifted from row 24-29 down to 18-23? No, looking at the same columns.
        # This looks like a movement of a 6x6 block.
        # We need to find the object and move it.
        # Let's identify the "active" object (color 6).
        # Find all coordinates of color 6.
        coords = np.argwhere(grid == 6)
        if len(coords) > 0:
            min_r, min_c = coords.min(axis=0)
            max_r, max_c = coords.max(axis=0)
            # Try moving it up by some offset.
            # Shift the area around the object.
            # For example, if we are shifting a 6x6 square.
            #<|channel>thought
            # In ACTION1, the change is r18c11:6x6... which means row 18 col 11 becomes 6x6.
            # In ACTION2, the change is r24c11:5x2,3x2,5x2...
            # It seems action 1 moves something UP.
            # Action 2 moves something DOWN.
            # Action 3 moves something LEFT/RIGHT?
            # Action 4 might be toggling or interacting with other blocks.
            pass

    # Since the exact movement logic is complex to induce from few deltas,
    # and the goal is a simple general rule, let's look at the patterns.
    # The transitions show that color 6 (the "player") moves in steps of 6 rows.
    # Let's implement a basic shift for the player block.
    
    # Find the bounding box of the 'player' (color 6).
    coords = np.argwhere(grid == 6)
    if len(coords) > 0:
        min_r, min_c = coords.min(axis=0)
        max_r, max_c = coords.max(axis=0)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        
        if action == 1: # Move Up
            dr, dc = -6, 0
        elif action == 2: # Move Down
            dr, dc = 6, 0
        elif action == 3: # Move Left/Right? ACTION3 delta shows r19c36... which is far right.
            # Action 3 seems to move something horizontally or toggle blocks.
            dr, dc = 0, 6
        elif action == 4: # Interaction / Toggle
            # Action 4 changes small areas (2x2 or similar).
            return grid # For now, keep it simple.
        else:
            dr, dc = 0, 0
            
        # Apply movement if not out of bounds
        if dr != 0 or dc != 0:
            # Clear old position
            new_grid[min_r:max_r+1, min_c:max_c+1] = 5 # Background color
            # Set new position
            nr, nc = min_r + dr, min_c + dc
            if 0 <= nr < H and 0 <= nc < W and 0 <= nr+h < H and 0 <= nc+w < W:
                new_grid[nr:nr+h, nc:nc+w] = 6
    
    return new_grid

def is_level_complete(grid):
    """
    The win state is usually when the player reaches a target area or collects all items.
    Given the data, we don't have a WIN STATE grid, but typically it involves
    the player (color 6) overlapping with a specific target color or region.
    """
    # Check if any cell of color 6 is in a 'target' zone (e.g., bottom rows).
    coords = np.argwhere(grid == 6)
    if len(coords) > 0:
        # If the player has reached the bottom section (rows 53-63), it might be complete.
        if coords.max(axis=0)[0] >= 53:
            return True
    return False