import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape)
    # Action mapping based on observed transitions:
    # ACTION 2: Down
    # ACTION 3: Left
    # ACTION 4: Right
    # ACTION 1: Up (inferred)
    
    # Find all "movable" objects (color 14 and color 4/5/0 etc inside blocks)
    # We need to identify the player-controlled entity.
    # In these levels, it seems there's a a set of small objects that move together or independently.
    # Looking at the deltas, the 'active' objects are those moving in the same direction.
    # Let's define the "player" as any connected component of color 14.
    # 
    # The observed movements:
    # ACTION 4 (Right): r30c18:1x3,14x3 -> shifted right? No, let's look closer.
    # INITIAL: obj6 (color 14) is bbox=(30, 18, 32, 20), pixels=8.
    # After ACTION 4: r30c18:1x3,14x3... this means cells (30,18),(30,19),(30,20) become 1, and (30,21),(30,22),(30,23) become 14.
    # This is a shift of +3 columns for the block of color 14.
    # Action 4 = Right (+3 cols), Action 3 = Left (-3 cols), Action 2 = Down (+3 rows), Action 1 = Up (-3 rows).
    
    # Identify all movable blocks (connected components of color 14)
    # We need to find the objects that are actually moving.
    # In the transitions, only specific blocks move.
    # Let's assume any object of color 14 moves if it can.
    
    # Find all connected components of color 14
    # from scipy.ndimage import label
    from scipy.ndimage import label
    
    labeled_array, num_features = label(grid == 14)
    if num_features == 0:
        return grid.copy()

    # Map action to direction
    directions = {
        1: (-3, 0), # Up
        2: (3, 0),  # Down
        3: (0, -3), # Left
        4: (0, 3),  # Right
    }
    
    if action not in directions:
        return grid.copy()
    
    dr, dc = directions[action]
    new_grid = grid.copy()
    
    # For each component, try to move it
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == i)
        
        # Check if destination is clear (color 2 or background)
        # We need to check the same shift for all pixels of the object
        can_move = True
        for r, c in coords:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                continue
            # In this game, "clear" might be color 2 (the main background)
            # if new_grid[nr, nc] != 2:
                # can_move = False
                # break
            pass
        
        # Move the object
        # Create a mask of the object
        obj_mask = (labeled_array == i)
        
        # Shift the object's current position to its own previous position
        # To avoid overwriting other objects, we first set old positions to background (color 2)
        # Let's refine this movement logic:
        # The observed deltas show that the block moves AND the cells it leaves behind are filled with color 1.
        # The same happens when moving into color 14 blocks.
    
    # Simplified approach based on observed transitions:
    # ACTION 4: r30c18:1x3,14x3 -> shifted right by 3. Old pixels became color 1.
    # ACTION 3: r30c21:14x3,1x3 -> shifted left by 3. Old pixels became color 1.
    # ACTION 2: r30c21:1x3... r33c21:14x3 -> shifted down by 3.
    
    # Find all components of color 14
    # find all connected components of color 14
    from scipy.ndimage import label as ndlabel
    labeled_array, num_features = ndlabel(grid == 14)
    if num_features == 0:
        return grid.copy()

    # Map action to direction
    directions = {
        1: (-3, 0), # Up
        2: (3, 0),  # Down
        3: (0, -3), # Left
        4: (0, 3),  # Right
    }
    
    if action not in directions:
        return grid.copy()
    
    dr, dc = directions[action]
    new_grid = grid.copy()
    
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == i)
        
        # Move the object
        # old_pos = coords
        # new_pos = coords + [dr, dc]
        
        # Shift pixels to background/color 1
        for r, c in coords:
            new_grid[r, c] = 1 if grid[r, c] == 14 else grid[r, c]
            
        for r, c in coords:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                new_grid[nr, nc] = 14
            else:
                # Object might be destroyed or clipped
                pass

    return new_grid

def is_level_complete(grid):
    # Based on observed transitions, there's no win state provided.
    # But usually it's about reaching a target or collecting something.
    # In this game, color 4 (the bottom line) is being modified?
    # "r63c63:0x1", "r63c62:0x1" etc.
    # This means the cells at the very bottom right are changing from 4 to 0.
    # The level is complete when that bottom row is filled with 0s? Or some specific cell is reached.
    # Let's assume it's complete when the bottom row contains any 0.
    # return np.any(grid[63, :] == 0)
    # Actually, let's look at the deltas again: r63c63:0x1, r63c62:0x1...
    # These happen every time ACTION 4/3/2 is performed.
    # It's like a progress bar.
    # return False # No clear win condition in data.
    return False