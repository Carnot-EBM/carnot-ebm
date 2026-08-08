import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state based on the current grid and action.
    
    The game involves two player objects (color 4 and color 5) and a goal object (color 5 at x=63).
    - ACTION2: Moves player objects DOWN.
    - ACTION3: Moves player object 5 LEFT, player object 4 RIGHT, and the goal object DOWN.
    """
    new_grid = grid.copy()
    
    # Identify player objects and the goal object
    # obj5: color 5, not at the bottom strip (y < 63)
    # obj6: color 4
    # obj3: color 5, at the far right edge (x == 63)
    obj5_mask = (grid == 5) & (np.arange(grid.shape[0])[:, None] < 63)
    obj6_mask = (grid == 4)
    obj3_mask = (grid == 5) & (np.arange(grid.shape[1])[None, :] == 63)
    
    # Clear old positions of moving objects to avoid duplication
    # We use color 9 as the default background fill
    new_grid[obj5_mask] = 9
    new_grid[obj6_mask] = 9
    new_grid[obj3_mask] = 9
    
    if action == 2:
        # ACTION2: Move player objects DOWN by 3, goal object DOWN by 1
        # Move obj5
        y, x = np.where(obj5_mask)
        for ny, nx in zip(y + 3, x):
            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                new_grid[ny, nx] = 5
        # Move obj6
        y, x = np.where(obj6_mask)
        for ny, nx in zip(y + 3, x):
            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                new_grid[ny, nx] = 4
        # Move obj3
        y, x = np.where(obj3_mask)
        for ny, nx in zip(y + 1, x):
            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                new_grid[ny, nx] = 5
                
    elif action == 3:
        # ACTION3: Move obj5 LEFT by 3, obj6 RIGHT by 3, goal object DOWN by 1
        # Move obj5
        y, x = np.where(obj5_mask)
        for ny, nx in zip(y, x - 3):
            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                new_grid[ny, nx] = 5
        # Move obj6
        y, x = np.where(obj6_mask)
        for ny, nx in zip(y, x + 3):
            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                new_grid[ny, nx] = 4
        # Move obj3
        y, x = np.where(obj3_mask)
        for ny, nx in zip(y + 1, x):
            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                new_grid[ny, nx] = 5
                
    return new_grid

def is_level_complete(grid):
    """
    Returns True if the level is complete.
    The level is completed when the color 4 object (obj6) covers the area 
    where the color 11 object was initially located (approx. y: 45-53, x: 51-59).
    """
    # Check if any cell of color 4 is within the target win zone
    return np.any(grid[45:54, 51:60] == 4)