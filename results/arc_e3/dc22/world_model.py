import numpy as np

def engine(grid, action, data):
    """
    Applies the observed transition rules to the grid.
    
    Rules inferred from the data:
    1. ACTION4: Toggles specific cells in rows 32, 33, and 63.
       - r32c16, r33c16, r32c18, r33c18 are toggled.
       - r63c6 is toggled.
    2. ACTION1: Toggles specific cells in rows 30, 31, 32, 33.
       - r30c20, r31c20, r32c20, r33c20 are toggled.
    3. ACTION6: Clicks at specific pixel coordinates (logical coords).
       - Clicks toggle cells in a pattern around the click location.
       - The pattern seems to be a 3x3 or similar area centered or offset from the click.
       - Based on the deltas, it toggles cells in a specific shape.
    
    Since the exact logic of ACTION6 is complex and depends on the click position,
    we will implement a simplified version that matches the observed deltas.
    However, given the complexity, we will focus on the deterministic toggling for ACTION4 and ACTION1,
    and for ACTION6, we will assume it toggles cells in a specific pattern based on the click coordinates.
    
    Note: The observed deltas for ACTION6 are complex and might involve a specific pattern.
    Without more data, we will implement a basic toggle for ACTION6 based on the click coordinates.
    """
    
    # Convert grid to a mutable copy
    new_grid = grid.copy()
    
    if action == 4:
        # Toggle specific cells
        # r32c16, r33c16, r32c18, r33c18, r63c6
        new_grid[32, 16] = 1 - new_grid[32, 16]
        new_grid[33, 16] = 1 - new_grid[33, 16]
        new_grid[32, 18] = 1 - new_grid[32, 18]
        new_grid[33, 18] = 1 - new_grid[33, 18]
        new_grid[63, 6] = 1 - new_grid[63, 6]
        
    elif action == 1:
        # Toggle specific cells
        # r30c20, r31c20, r32c20, r33c20
        new_grid[30, 20] = 1 - new_grid[30, 20]
        new_grid[31, 20] = 1 - new_grid[31, 20]
        new_grid[32, 20] = 1 - new_grid[32, 20]
        new_grid[33, 20] = 1 - new_grid[33, 20]
        
    elif action == 6:
        # Click action
        if data is not None:
            x, y = data['x'], data['y']
            # Toggle cells in a specific pattern based on the click coordinates
            # Based on the observed deltas, we will toggle cells in a 3x3 area centered at (x, y)
            # However, the observed deltas are more complex, so we will implement a basic toggle
            # for the cells around the click location.
            # Note: The observed deltas for ACTION6 are complex and might involve a specific pattern.
            # Without more data, we will implement a basic toggle for ACTION6 based on the click coordinates.
            # We will toggle cells in a 3x3 area centered at (x, y)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 64 and 0 <= ny < 64:
                        new_grid[nx, ny] = 1 - new_grid[nx, ny]
    
    return new_grid

def is_level_complete(grid):
    """
    Checks if the grid is in a level-complete state.
    
    Based on the observed win state, the level is complete when:
    - Rows 0-2, 56-62 are all color 3.
    - Row 63 is all color 0.
    - Other rows have specific patterns, but the key indicator is the presence of color 0 in row 63.
    """
    # Check if row 63 is all color 0
    if not np.all(grid[63] == 0):
        return False
    
    # Check if rows 0-2, 56-62 are all color 3
    for i in range(3):
        if not np.all(grid[i] == 3):
            return False
    for i in range(56, 63):
        if not np.all(grid[i] == 3):
            return False
    
    return True