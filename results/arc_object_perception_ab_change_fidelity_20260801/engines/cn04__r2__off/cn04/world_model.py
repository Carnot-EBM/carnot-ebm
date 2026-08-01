import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state based on the observed transitions.
    ACTION 4 (Right) moves a specific pattern of cells.
    The pattern consists of several vertical bars and gaps.
    """
    if action == 4:
        # The movement is a shift to the right by 1 logical unit.
        # We need to identify which parts of the same object are moving.
        # Only certain colors/patterns move.
        # In this case, it's a simple translation of all non-background color 10.
        # Note: ACTION 4 in these games often corresponds to 'right'.
        
        new_grid = grid.copy()
        # Find all cells that are not background (color 10)
        # Background is 10.
        # Create a mask for everything that isn't 10.
        mask = (grid != 10)
        
        # Shift the mask and values to the right by 1.
        # We must handle the boundary carefully.
        shifted_mask = np.roll(mask, 1, axis=1)
        shifted_values = np.roll(grid, 1, axis=1)
        
        # Apply the shifted values where the mask was true.
        # result[r, c+1] = grid[r, c]
        # new_grid[r, c+1] = grid[r, c]
        #<|channel>thought
        # The observed transitions show that as the object moves right, 
        # it replaces the background (10).
        # However, if it hits something or moves into a space, 
        # it doesn't just roll; it shifts.
        
        # Let's refine: only move non-background pixels.
        # For each row, find indices of non-10 pixels.
        for r in range(grid.shape[0]):
            row = grid[r, :]
            non_bg_indices = np.where(row != 10)[0]
            if len(non_bg_indices) > 0:
                # Shift these specific indices to the right by 1.
                # To avoid overwriting, we work from right to left.
                for c in reversed(non_bg_indices):
                    val = grid[r, c]
                    if c + 1 < grid.shape[1]:
                        new_grid[r, c+1] = val
                    # The original cell becomes background unless another pixel moves into it.
                    # But wait, the delta shows cells changing TO 10 and others FROM 10.
                    # This is a translation.
                    
        # Now we need to clear the trailing edges.
        # In a simple translation, if you shift everything right, 
        # the leftmost column of the object becomes background.
        for r in range(grid.shape[0]):
            row = grid[r, :]
            non_bg_indices = np.where(row != 10)[0]
            if len(non_bg_indices) > 0:
                leftmost = non_bg_indices[0]
                new_grid[r, leftmost] = 10
                
        return new_grid

    return grid

def is_level_complete(grid):
    """
    Determines if the level is complete based on the win state.
    Usually, this involves reaching a certain configuration or clearing objects.
    Given no WIN STATE grid was provided, we look for common patterns.
    In many ARC games, completion is when an object reaches a target area.
    Since we don't have the target, we return False unless a specific condition is met.
    """
    # Without a win state example, it's hard to be precise.
    # However, often 'is_level_complete' is checked after every action.
    # We'll assume for now that there isn't a simple color-based win condition.
    return False