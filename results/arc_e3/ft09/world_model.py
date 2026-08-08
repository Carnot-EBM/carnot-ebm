import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state based on the action and data.
    The game involves clicking 6x6 blocks to change their color to 8.
    Specific clicks also trigger changes in the bottom row (row 63).
    """
    grid = grid.copy()
    if action == 6:
        # ACTION6 is a click with data={'x': px, 'y': py}
        # x is column, y is row
        x, y = data['x'], data['y']
        
        # Change the 6x6 block starting at (y, x) to color 8
        # Ensure the block is within grid boundaries
        y_end = min(y + 6, grid.shape[0])
        x_end = min(x + 6, grid.shape[1])
        grid[y:y_end, x:x_end] = 8
        
        # Specific clicks trigger changes in the bottom row (row 63)
        # These are observed in the transition deltas
        if x == 36 and y == 36:
            # Click 1: r63c62:11x2
            grid[63, 62:64] = 11
        elif x == 36 and y == 44:
            # Click 2: r63c60:11x2
            grid[63, 60:62] = 11
        elif x == 52 and y == 44:
            # Click 3: r63c58:11x2
            grid[63, 58:60] = 11
        elif x == 36 and y == 52:
            # Click 4: r63c58:12x6 (resets the bottom row segment to color 12)
            grid[63, 58:64] = 12
            
    return grid

def is_level_complete(grid):
    """
    Returns True if the level is complete.
    The level is complete when four specific 6x6 blocks are all color 8.
    These blocks form a T-shape in the bottom-right quadrant.
    """
    # The four 6x6 blocks identified from the observed transitions:
    # Block 1: Row 36, Col 36
    # Block 2: Row 44, Col 36
    # Block 3: Row 44, Col 52
    # Block 4: Row 52, Col 36
    target_blocks = [
        (36, 36),
        (44, 36),
        (44, 52),
        (52, 36)
    ]
    
    for r, c in target_blocks:
        # Check if the 6x6 block starting at (r, c) is entirely color 8
        if not np.all(grid[r:r+6, c:c+6] == 8):
            return False
            
    return True