def engine(grid, action, data):
    """
    World model engine for ARC task tu93.
    
    The grid appears to represent a 2D space where objects move based on actions.
    Actions seem to correspond to directions:
    0: up, 1: right, 2: down, 3: left, 4: special
    
    Based on the failing cases, we need to understand the movement patterns.
    """
    import numpy as np
    
    # Convert grid to numpy array for easier manipulation
    g = np.array(grid)
    rows, cols = g.shape
    
    # Create a copy to modify
    new_g = g.copy()
    
    # Action mapping: 0=up, 1=right, 2=down, 3=left, 4=special
    # Based on the failing cases, let's analyze the patterns
    
    if action == 3:  # Left
        # Case 0: [63, 63, 6, 0] -> [63, 63, 6, 0]
        # Case 4: [63, 58, 6, 0] -> [63, 58, 6, 0]
        # These seem to be single-cell changes where the value changes
        # Looking at the pattern, it seems like when action is 3 (left),
        # certain cells change their values
        
        # Find all non-zero cells
        non_zero = np.argwhere(g != 0)
        
        for r, c in non_zero:
            # Check if this cell should change based on action 3
            # From the examples, it seems like specific cells change
            # Let's look for a pattern in the coordinates and values
            
            # It appears that when action is 3, cells with certain properties change
            # The change seems to be related to the value 6 appearing
            if g[r, c] == 6:
                # When there's a 6, it might be moving or changing
                pass
    
    elif action == 4:  # Special
        # Case 1: Multiple cells change from 0 to 9 or from 9 to 0
        # The pattern shows a 3x3 area changing
        # Cells at (15,15), (15,16), (15,17), (15,21), (15,22), (15,23), (16,15), (16,16)
        # change from 0 to 9, and some from 9 to 0
        
        # This looks like a pattern where a 3x3 block is being activated/deactivated
        # The pattern seems to be related to the position of existing non-zero cells
        
        # Find the bounding box of non-zero cells
        non_zero = np.argwhere(g != 0)
        if len(non_zero) > 0:
            min_r, min_c = non_zero.min(axis=0)
            max_r, max_c = non_zero.max(axis=0)
            
            # The changes seem to occur in a specific pattern
            # Let's look for a 3x3 pattern around certain points
            pass
    
    elif action == 2:  # Down
        # Cases 2, 3: Similar patterns to action 4 but with different coordinates
        # The pattern shows 3x3 areas changing
        pass
    
    # Return the new grid
    return new_g.tolist()

def is_level_complete(grid):
    """
    Check if the level is complete.
    """
    import numpy as np
    g = np.array(grid)
    # A level is complete when there are no more changes possible
    # or when a specific condition is met
    return False