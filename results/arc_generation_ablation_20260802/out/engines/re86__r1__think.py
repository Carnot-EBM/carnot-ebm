import numpy as np

def engine(grid, action, data):
    """
    Induces the world model for game 're86'.
    Based on observed transitions:
    ACTION4 seems to be a rightward shift of a vertical column/structure.
    ACTION5 seems to trigger a state change or specific cell toggle.
    ACTION1 seems to be an upward shift of structures.
    The grid contains various colors representing walls (5), blocks (4, 9, 11, 15, 0).
    """
    new_grid = grid.copy()
    h, w = new_grid.shape

    if action == 4:
        # Rightward movement logic
        # Identify columns that are shifting based on delta patterns
        # The deltas show shifts at c39->c42->c45... and changes in row 24
        # We simulate a simple horizontal slide for the active elements
        for r in range(h):
            row = new_grid[r]
            # Find indices of non-background cells (assuming 5 is background)
            mask = row != 5
            if not np.any(mask): continue
            
            # Shift mask right by 3 pixels if it's part of the moving structure
            # Based on ACTION4 deltas: r11-r37 shifted from c39 to c42 then c45
            if 11 <= r <= 37:
                # This is a simplification; we move all non-5 cells in these rows right by 3
                shifted_row = np.full(w, 5, dtype=int)
                indices = np.where(row != 5)[0]
                for idx in indices:
                    if idx + 3 < w:
                        shifted_row[idx + 3] = row[idx]
                    else:
                        shifted_row[idx] = row[idx] # boundary
                new_grid[r] = shifted_row

    elif action == 1:
        # Upward movement logic
        # The deltas show shifts in vertical positions (e.g., r13->r10->r7->r4)
        # and changes in horizontal blocks (r24->r21->r18->r15)
        # We simulate an upward shift of active elements by 3 units
        for c in range(w):
            col = new_grid[:, c]
            mask = col != 5
            if not np.any(mask): continue
            
            shifted_col = np.full(h, 5, dtype=int)
            indices = np.where(col != 5)[0]
            for idx in indices:
                if idx - 3 >= 0:
                    shifted_col[idx - 3] = col[idx]
                else:
                    shifted_col[idx] = col[idx] # boundary
            new_grid[:, c] = shifted_col

    elif action == 5:
        # ACTION5 triggers specific cell updates based on the delta
        # Example: r24c48 becomes 9, r27c21 becomes 0
        # This looks like a "trigger" or "toggle" mechanism.
        # Since we don't have a general rule, we apply a minimal state change if possible.
        pass

    return new_grid

def is_level_complete(grid):
    """
    Determines if the win state is reached.
    Usually involves reaching a target position or clearing blocks.
    Based on the provided data, there isn't a clear WIN STATE grid, 
    but typically it's when certain colors reach specific coordinates.
    """
    # Check for common ARC-AGI win conditions (e.g., color 1 at bottom right)
    # In this case, we check if any 'goal' cells are filled.
    if np.any(grid[63, 59:] != 5):
        return True
    return False