import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, ACTION3 moves a specific object (a small 2x4 cluster of colors [9, 10, 2, 2])
    # and potentially changes some background elements (color 14 at col 62-63).
    # The object consists of values [9, 10, 2, 2] in rows 19, 20.
    # The object's width is 4 columns.
    
    next_grid = grid.copy()
    
    if action == 3: # Move Left
        # Find the target object (the 2x4 pattern)
        # We look for the same pattern that shifted left by 2 units per transition.
        # Look for the first occurrence of color 9 in row 19.
        obj_pos = None
        for c in range(grid.shape[1]):
            if grid[19, c] == 9:
                obj_pos = c
                break
        
        if obj_pos is not None:
            # Define the object mask (relative to top-left corner)
            # Object shape is 2x4. Values are [9, 10, 2, 2].
            # Pattern is repeated in rows 19 and 20.
            # Original position was column 35.
            # 37 -> 35 -> 33 -> 31 -> 29 -> 27 -> 25 -> 23
            # Each ACTION3 moves it 2 pixels left.
            
            # Clear old position
            # The background at those positions should be restored to color 5.
            # next_grid[19, obj_pos : obj_pos + 4] = 5
            # next_grid[20, obj_pos : obj_pos + 4] = 5
            
            # Move object 2 units left
            new_pos = max(0, obj_pos - 2)
            
            # Restore background of trailing edge
            next_grid[19, obj_pos + 2 : obj_pos + 4] = 5
            next_grid[20, obj_pos + 2 : obj_pos + 4] = 5
            
            # Overwrite new position
            next_grid[19, new_pos : new_pos + 4] = [9, 10, 2, 2]
            next_grid[20, new_pos : new_pos + 4] = [9, 10, 2, 2]
            
        # Also handle the background changes (color 14 -> 0).
        # Since we observed that r6c62:0x2, r8c62:0x2 etc. occur in pairs or singles,
        # it's a}
    return next_grid

def is_level_complete(grid):
    # Based on the provided data, no win state was given.
    # return True if `grid` is a level_complete / win state, {// own logic based on grid pattern}.
    # The object has to reach some target area.
    # Check if the same object cluster has reached column 0.
    for c in range(grid.shape[1]):
        if grid[19, c] == 9:
            # If object is at col 0, assume complete.
            return c == 0
    return False