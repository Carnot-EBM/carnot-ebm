import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action 4: Right movement of a specific object/column
    if action == 4:
        # Find the target column containing color 9
        target_col = -1
        for c in range(grid.shape[1]):
            if any(grid[r, c] == 9 for r in range(grid.shape[0])):
                target_col = c
                break
        
        if target_col != -1:
            # Shift the column of color 9 to the right by 1
            new_grid = grid.copy()
            # We need to identify which cells are actually moving
            # To avoid overwriting, we shift everything that is part of the "object"
            # For each row, find if there's a 9 in that row
            for r in range(grid.shape[0]):
                if grid[r, target_col] == 9:
                    new_grid[r, target_col] = 5 # Reset old position to background
                    new_grid[r, target_col + 1] = 9
                elif grid[r, target_col+1] == 9:
                     # This handles cases where it might be shifted and already present
                    pass
            
            # Special case for ACTION4 delta shows shifting columns of 9s
            # In some rows, 24th row (index 23) has specific behavior
            # The object being moved is the vertical line of 9s.
            # Let's refine this move logic.
            
            # Find all coordinates of 9s
            coords_9 = np.argwhere(grid == 9)
            if len(coords_9) > 0:
                # Find the minimum column index among all 9s
                min_col = np.min(coords_9[:, 1])
                max_col = np.max(coords_9[:, 1])
                
                # Shift everything in that range
                for r, c in coords_9:
                    new_grid[r, c] = 5
                    if c + 1 < grid.shape[1]:
                        new_grid[r, c + 1] = 9
                
                # For special cells like r24c48 (from delta), we need to a few things
                # Specifically, it seems there are a couple of 9s moving together as aC-//-B structure
    
    # Simplified general rule based on observed ACTION4 shifts columns of color 9
    # Based on thes deltas, ACTION4 moves the 'column' of 9s one step right.
    # Let's just implement a shift for the same set of rows.
    
    # Since the rules are complex and specific, let's a very simple implementation
    # that mimics the movement of the target object.
    
    # We actually see Action 4 is Right, Action 1 is Up?
    # No, Action 1 is shifting something Up.
    # Wait, looking at the laout, 9 is a key color.
    # Let's try a morethoughtful approach.
    
    return grid # Placeholder if logic is too<|channel>thoughtfully’

def engine(grid, action, data):
    # The provided transitions show ACTION4 shifts a column of 9s to the right.
    # ACTION1 shifts some parts of the grid upwards.
    # The goal is likely to align colors or move objects.
    # Color 5 is background.
    
    new_grid = grid.copy()
    if action == 4: # Move Right
        coords_9 = np.argwhere(grid == 9)
        if len(coords_9) > 0:
            for r, c in coords_9:
                new_grid[r, c] = 5
                if c + 1 < grid.shape[1]:
                    new_grid[r, c+1] = 9
    elif action == 1: # Move Up (observed as shifting rows/cols)
        # Based on deltas, it seems to be moving blocks of pixels up.
        # We find the block of 11s and shift them.
        coords_11 = np.argwhere(grid == 11)
        if len(coords_11) > 0:
            for r, c in coords_11:
                new_grid[r, c] = 5
                if r - 3 >= 0:
                    new_grid[r-3, c] = 11
    elif action == 5: # Special trigger?
        pass

    return new_grid

def is_level_complete(grid):
    # The game ends when a certain condition is met.
    # In many ARC games, this is aligning colors or clearing a board.
    # Let's check if any color 9 has reached a target column.
    # Or simply return False for now since no win state was provided.
    return False