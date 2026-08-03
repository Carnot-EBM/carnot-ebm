import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION4: Right movement of a specific object/pattern.
    # ACTION5: Left movement or toggle?
    # ACTION1: Upward movement of a pattern.
    # Based on the same delta patterns, we look for a shift in coordinates.
    
    new_grid = grid.copy()
    
    if action == 4: # Move Right
        # Identify the vertical bar of color 9 at x=42 (initially) and move it right by 3 columns.
        # We also need to handle the other elements that are shifted.
        # The logic seems to be shifting a "column" of values.
        # Find all cells of color 9 that form a vertical line.
        # For each cell of color 9, if its row has only one 9, wes shift it.
        # Shift the column of 9s.
        mask_9 = (grid == 9)
        coords_9 = np.argwhere(mask_9)
        if coords_9.size > 0:
            # Group by column.
            cols_9 = np.unique(coords_9[:, 1])
            for col in cols_9:
                # Check if this is the 'movable' column of 9s.
                # If therethought process omitted as per rules.
                pass

    # Since inducing exact movement rules from sparse deltas is complex, 
    # let' same implement a simple coordinate-based shift for the observed patterns.
    
    # Let's try to find the specific object being moved.
    # In ACTION4, r11c39 becomes 5 and r11c42 becomes 9. This means something at c39 was replaced by 5 and something at c42 became 9.
    # initial grid: r11c39 is 9. So action 4 moves the column of 9s from c39 to c42? No, wait.
    # INITIAL GRID: r11: 5x39, 9x1, 5x24 -> Col 39 is color 9.
    # ACTION 4 (first): r11c39: 5x1, r11c42: 9x1. Column 39 becomes 5, Column 42 becomes 9.
    # Shift = +3 columns.
    
    if action == 4: # Move Right
        shift = 3
        mask_9 = (grid == 9)
        coords_9 = np.argwhere(mask_9)
        if coords_9.size > 0:
            # Find the most common column among cells of color 9.
            col_9 = coords_9[0, 1]
            new_grid[coords_9[:, 0], col_9] = 5
            new_grid[coords_9[:, 0], col_9 + shift] = 9
            
    elif action == 5: # Move Left
        shift = -3
        mask_9 = (grid == 9)
        coords_9 = np.argwhere(mask_9)
        if coords_9.size > 0:
            col_9 = coords_9[0, 1]
            new_grid[coords_9[:, 0], col_9] = 5
            new_grid[coords_9[:, 0], col_9 + shift] = 9

    elif action == 1: # Move Up
        # ACTION 1 deltas show r13c21 becomes 11 and r36c21 becomes 5.
        # This is a moving pattern of color 11.
        # Find all cells of color 11.
        # Shift them up by 3 rows.
        shift = -3
        mask_11 = (grid == 11)
        coords_11 = np.argwhere(mask_11)
        if coords_11.size > 0:
            # We move the whole block of 11s.
            for r, c in coords_11:
                new_grid[r + shift, c] = 11 if (r+shift >= 0) else grid[r+shift, c]
                new_grid[r, c] = 5
        # Note: this is very simplified. The observed deltas for Action 1 are complex.
        # Let's refine based on the laout.
        
    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a target or collecting something.
    # In these games, often it's when a specific cell reaches a certain value.
    # Check if any cell of color 9 has reached the far right.
    mask_9 = (grid == 9)
    coords_9 = np.argwhere(mask_9)
    if coords_9.size > 0:
        return coords_9[:, 1].max() >= 63
    return False