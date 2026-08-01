import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION4: Right movement of a specific object/pattern.
    # ACTION5: Left movement or toggle? Actually looks like it's shifting something right then left.
    # ACTION1: Upward shift of patterns.
    # Based on the same pattern of changes in ACTION4 and ACTION1, 
    # we can identify that there are "blocks" of colors moving together.
    
    new_grid = grid.copy()
    
    if action == 4: # Move Right
        # Find all cells that are not color 5 (background)
        # We look for the same relative positions of non-background cells.
        # The logic seems to be a simple translation of all non-background cells by (0, 1) if they aren't blocked.
        # Shift everything that isn't background to the right by 3 pixels.
        # Note: In the observed transitions, ACTION4 shifts things by 3 columns.
        shift_x = 3
        mask = (grid != 5)
        # To avoid overwriting, we create a temporary mask of what will move.
        # shifted_mask = np.roll(mask, shift_x, axis=1)
        # 
        #<|channel>thought: The observed delta is very specific. It's shifting a vertical bar of color 9 at col 42 to 45, then 48.
        # Let's implement a general shift for any cell that is not color 5.
        # For each row, find indices of non-background cells and move them.
        for r in range(grid.shape[0]):
            row = grid[r]
            non_bg = np.where(row != 5)[0]
            if len(non_bg) > 0:
                # Move every non-bg cell to the right by 3.
                # We must handle boundaries and potential collisions.
                # However, based on observations, it seems like a simple translation.
                pass

    # Since inducing a full physics engine from limited deltas is hard, 
    # let's look at the patterns again. ACTION4 shifts things by +3 columns.
    # ACTION1 shifts things by -3 rows (upwards).
    
    # Re-evaluating ACTION4: r11c39:5x1 r11c42:9x1 -> means cell (11, 39) becomes 5, (11, 42) becomes 9.
    # This is exactly a shift of +3 columns.
    
    # Re-evaluating ACTION1: r13c21:11x1 r14c21:11x1 ... r63c55:1x1
    # It looks like some objects move up by 3 units.
    
    # Let's implement a general "shift" for all cells not equal to background color 5.
    def shift_all(grid, dr, dc):
        res = np.full_like(grid, 5)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != 5:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                        res[nr, nc] = grid[r, c]
        return res

    if action == 4: # Right
        return shift_all(grid, 0, 3)
    elif action == 1: # Up
        return shift_all(grid, -3, 0)
    elif action == 5: # Left (implied opposite of 4)
        return shift_all(grid, 0, -3)
    
    return new_grid

def is_level_complete(grid):
    # The win state isn't explicitly given as a full grid, but usually involves 
    # reaching a certain configuration or collecting items.
    # In this game, the bottom row has some special values.
    # Let's assume it's complete if the bottom row contains only color 15 or something similar.
    # Looking at INITIAL GRID r63: 15x59, 1x5.
    # After ACTION4, r63c58 becomes 1x1. This looks like a progress bar.
    # If the "1" reaches the end of the row (col 63), it might be complete.
    return np.any(grid[63, :] == 1) and grid[63, 63] == 1