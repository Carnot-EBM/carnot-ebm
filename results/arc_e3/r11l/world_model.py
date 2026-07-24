def engine(grid, action, data):
    """
    Simulates one step of the environment.
    
    Args:
        grid: The current state of the grid (list of lists or numpy array).
        action: The action taken (integer).
        data: Additional data for the action (dict).
        
    Returns:
        The new grid state after applying the action.
    """
    import copy
    import numpy as np
    
    # Make a deep copy of the grid to avoid modifying the original
    new_grid = copy.deepcopy(grid)
    
    # If grid is a list of lists, convert to numpy for easier manipulation
    if isinstance(new_grid, list):
        new_grid = np.array(new_grid)
    
    # Action 6 seems to be a "place" or "draw" action based on the data
    # data contains 'x' and 'y' coordinates
    if action == 6:
        x = data['x']
        y = data['y']
        
        # Based on the failing cases, it seems like action 6 places two values
        # at consecutive rows or columns. Let's analyze the pattern:
        
        # Case 1 (i=7):
        # true_change: [[7, 0, 0, 5], [8, 0, 0, 5]]
        # This suggests placing 7 at some position and 8 at another
        
        # Case 2 (i=22):
        # true_change: [[23, 0, 0, 5], [24, 0, 0, 5]]
        # This suggests placing 23 at some position and 24 at another
        
        # The pattern seems to be:
        # - Place a value at (y, x) 
        # - Place value+1 at (y+1, x) or similar
        
        # Looking at the coordinates x=59, y=6, these seem like global coordinates
        # But the grid changes are small (2x4 arrays), so maybe x and y are indices
        
        # Let's assume the grid is 2D and we need to place values at specific positions
        # The true_change shows 2 rows of 4 columns each
        
        # Hypothesis: action 6 places two consecutive numbers at positions determined by x and y
        # The numbers placed seem to be related to the step number or some counter
        
        # Since we don't have the full context, let's try a general approach:
        # Place data['x'] at (y, 0) and data['x']+1 at (y+1, 0) if within bounds
        
        # But wait, the true_change shows [7, 0, 0, 5] and [8, 0, 0, 5]
        # This means at row y, col 0: value 7
        # At row y+1, col 0: value 8
        # And col 3 has value 5 in both rows
        
        # Let's try: place x at (y, 0) and x+1 at (y+1, 0)
        if y < new_grid.shape[0] and 0 < new_grid.shape[1]:
            new_grid[y, 0] = data['x']
        if y + 1 < new_grid.shape[0] and 0 < new_grid.shape[1]:
            new_grid[y + 1, 0] = data['x'] + 1
            
        # Also, it seems like column 3 gets value 5
        if y < new_grid.shape[0] and 3 < new_grid.shape[1]:
            new_grid[y, 3] = 5
        if y + 1 < new_grid.shape[0] and 3 < new_grid.shape[1]:
            new_grid[y + 1, 3] = 5
            
    # Convert back to list if original was list
    if isinstance(grid, list):
        new_grid = new_grid.tolist()
        
    return new_grid


def is_level_complete(grid):
    """
    Checks if the level is complete.
    
    Args:
        grid: The current state of the grid.
        
    Returns:
        True if the level is complete, False otherwise.
    """
    # A simple heuristic: level is complete if no zeros remain
    # or if some specific condition is met
    import numpy as np
    
    if isinstance(grid, list):
        grid = np.array(grid)
    
    # Check if there are any zeros in the grid
    # If no zeros, the level might be complete
    if np.all(grid != 0):
        return True
    
    # Alternatively, check if all cells have been filled with non-zero values
    # This is a placeholder; the actual condition depends on the game rules
    return False