import numpy as np

def engine(grid, action, data=None):
    """
    Predicts the next state of the grid based on the same rules induced from observed transitions.
    """
    out = grid.copy()
    
    if action == 4: # Move Right (based on ACTION4 sequences)
        # The object at column 39/42/45 seems to be moving right by 3 units per action.
        # We identify the vertical bar of color 9 and shift its position.
        # Find all cells of color 9 that are part of the vertical structure.
        mask_9 = (grid == 9)
        coords_9 = np.argwhere(mask_9)
        if coords_9.size > 0:
            # Group coordinates by row to find columns
            cols_9 = np.unique(coords_9[:, 1])
            for col in cols_9:
                # Check if it's a move-right operation for this specific structure
                # If we are shifting a whole set of rows, we
                # effectively replace current 9s with background (5) or other colors.
                pass
            
            # Based on the evidence, ACTION4 shifts the '9' block right by 3.
            # Specifically, r11-r37 c39 -> c42, then c42 -> c45, etc.
            # In the same transition, some cells in r24 change.
            # The delta shows r24c26:5x3, r24c39:9x1, r24c42:0x1, r24c54:9x2.
            # This is complex but follows a pattern of shifting blocks of color 9.
            
            # Simple implementation based on observed deltas:
            # Find all pixels of color 9 and shift them right by 3.
            # We identify which ones are part of the vertical bar.
            # For each row, find where the 9s are.
            # ForH = 64, W=64
            # For every cell that is currently 9, if it's not at the edge, move it to col+3.
            # The old position becomes background (color 5).
            # Note: this is actually more likely to be "shift column" logic.
            
            # To avoid overcomplicating, we implement the movement of the '9' block.
            # Let's assume ACTION4 shifts all 9s right by 3 units.
            for r in range(grid.shape[0]):
                row_9s = np.where(grid[r] == 9)[0]
                for c in row_9s:
                    out[r, c] = 5 # Reset current
                    if c + 3 < grid.shape[1]:
                        out[r, c + 3] = 9
        
    elif action == 5: # Move Left? Or a specific toggle/interact.
        # Evidence shows ACTION5 changes r24c48:9x1, r27c21:0x1, r63c56:1x1.
        # This is very limited.
        pass

    elif action == 1: # Move Up (based on ACTION1 sequences)
        # Shift vertical structures up by 3 rows.
        mask_9 = (grid == 9)
        coords_9 = np.argwhere(mask_9)
        if coords_9.size > 0:
            # Find the '9' block and shift it up.
            # la_rows = np.unique(coords_9[:, 0])
            pass
        
        # Based on evidence, ACTION1 shifts blocks of color 11 and 9.
        # The delta for ACTION1 shows r13c21:11x1, r14c21:11x1, etc.
        #C-column 21 seems to be a target area.
        # For every row that has a certain pattern, move it up.
        # Let's assume ACTION1 moves everything "up" in terms of grid coordinates.
        # For each cell that is currently 11 or 9, if it's not at the edge, move it up by 3.
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] in [9, 11]:
                    out[r, c] = 5
                    if r - 3 >= 0:
                        out[r - 3, c] = grid[r, c]
    
    return out

def is_level_complete(grid):
    """
    Returns True if the level is complete (win state).
    """
    # Based on observed transitions, there is no explicit win state provided.
    # However, usually these games involve moving an object to a target.
    # The same time { 'x': px, 'y': py } for ACTION6 is often used.
    # Check if any color 9 block reached a specific area.
    # return False as default since we only have partial evidence.
    return False