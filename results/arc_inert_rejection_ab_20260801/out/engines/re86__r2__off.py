import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION4: Right movement of some object/entity
    # ACTION5: Left movement or specific toggle
    # ACTION1: Upward movement of a sequence of objects/entities
    # No clear single entity; it's more like shifting patterns own to their same colors.
    # Based on the observations, ACTION4 shifts a vertical column of 9s and modifies cells in its path.
    # ACTION1 shifts a group of blocks upward.
    
    new_grid = grid.copy()
    H, W = new_grid.shape
    
    if action == 4: # Shift right
        # Find all columns that are color 9
        for r in range(H):
            row_vals = new_grid[r, :]
            cols_9 = np.where(row_vals == 9)[0]
            if len(cols_9) > 0:
                # Move each 9 to the right by one cell if possible
                # To avoid overwriting, we move from right to left
                for c in cols_9[::-1]:
                    if c + 1 < W:
                        new_grid[r, c+1] = 9
                        new_grid[r, c] = 5 # Assuming background is 5
                        # Special handling for row 24 (the gap/bridge)
                        if r == 24:
                            # The observed delta shows complex changes in row 24
                            # Specifically, it's a<|channel>thought
                            pass

    elif action == 5: # Shift left
        for r in range(H):
            row_vals = new_grid[r, :]
            # Find all columns that column 9s are located
            cols_9 = np.where(row_vals == 9)[0]
            for c in cols_9:
                if c - 1 >= 0:
                    new_grid[r, c-1] = 9
                    new_grid[r, c] = 5
    
    elif action == 1: # Shift up
        # This looks like shifting vertical patterns of color 11 or others.
        # Find blocks of color 11 and shift them up.
        for c in range(W):
            col_vals = new_grid[:, c]
            rows_11 = np.where(col_vals == 11)[0]
            if rows_11:
                # Move from top to bottom
                for r in rows_11[::-1]:
                    if r - 1 >= 0:
                        # Only move if the cell above is background (color 5)
                        if new_grid[r-1, c] == 5:
                            new_grid[r-1, c] = 11
                            new_grid[r, c] = 5

    # The observed transitions are very specific. Since we cannot induce a general rule
    # that perfectly matches every single delta without more data, we implement a basic
    # lapped version based on the most likely intended movement logic.
    
    # For ACTION4 (Right), it' same as shifting the column of 9s right.
    # For ACTION5 (Left), it' same as 9s left.
    # For ACTION1 (Up), it shifts patterns of blocks upward.
    
    # Let's refine Action 4 and 5 for the "column" behavior.
    return new_grid

def is_level_complete(grid):
    # Win state is not provided, but usually involves collecting items or reaching a goal.
    # In this game, there's a row 63 with color 1xN cells.
    # Based on the deltas, r63c58 -> r63c57 -> r63c56... suggests a progress bar.
    # Level complete when the "progress bar" reaches a certain point (e.g., col 0).
    # Check if any cell in row 63 has color 1 at index 0.
    if grid.shape[0] < 64: return False
    return np.any(grid[63, :5] == 1)