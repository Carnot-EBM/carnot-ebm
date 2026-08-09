import numpy as np

def engine(grid, action, data):
    """
    Induces the world model for ARC-AGI game 's5i5'.
    The game involves two control objects (color 2) that expand a horizontal block (color 14)
    and a vertical block (color 11). The level is completed when both blocks reach their targets.
    """
    if action != 6:
        return grid
    
    x, y = data['x'], data['y']
    new_grid = grid.copy()
    
    # Control Object 1: Bbox=(18, 36, 24, 48), Color=2
    # Clicking this expands Block 1 (Color 14) to the right by 3 pixels in rows 9, 10, 11.
    if 36 <= x <= 48 and 18 <= y <= 24:
        for r in [9, 10, 11]:
            cols = np.where(grid[r] == 14)[0]
            if len(cols) > 0:
                rightmost = cols[-1]
                for c in range(rightmost + 1, min(rightmost + 4, 64)):
                    new_grid[r, c] = 14
            else:
                # Initial position if not yet present
                for c in range(28, min(28+3, 64)):
                    new_grid[r, c] = 14
        
        # Cursor on row 63 grows leftward as a side effect
        cursor_cols = np.where(grid[63] == 4)[0]
        if len(cursor_cols) > 0:
            leftmost = cursor_cols[0]
            new_grid[63, max(0, leftmost - 1)] = 4
        elif grid[63, 63] == 4: # Start from the right edge
            new_grid[63, 62] = 4

    # Control Object 2: Bbox=(35, 21, 47, 27), Color=2
    # Clicking this expands Block 2 (Color 11) downwards by 3 pixels in columns 9, 10, 11.
    elif 21 <= x <= 27 and 35 <= y <= 47:
        for c in [9, 10, 11]:
            rows = np.where(grid[:, c] == 11)[0]
            if len(rows) > 0:
                bottommost = rows[-1]
                for r in range(bottommost + 1, min(bottommost + 4, 64)):
                    new_grid[r, c] = 11
            else:
                # Initial position if not yet present
                for r in range(28, min(28+3, 64)):
                    new_grid[r, c] = 11
        
        # Cursor on row 63 grows leftward as a side effect
        cursor_cols = np.where(grid[63] == 4)[0]
        if len(cursor_cols) > 0:
            leftmost = cursor_cols[0]
            new_grid[63, max(0, leftmost - 1)] = 4
        elif grid[63, 63] == 4: # Start from the right edge
            new_grid[63, 62] = 4

    return new_grid

def is_level_complete(grid):
    """
    The level is complete when Block 1 (color 14) reaches its target at col 52
    and Block 2 (color 11) reaches its target at row 51.
    """
    block1_reached = False
    for r in [9, 10, 11]:
        if np.any(grid[r, 52:] == 14):
            block1_reached = True
            break
            
    block2_reached = False
    for c in [9, 10, 11]:
        if np.any(grid[51:, c] == 11):
            block2_reached = True
            break
            
    return block1_reached and block2_reached