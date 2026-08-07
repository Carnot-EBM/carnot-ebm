import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION1: Move an active element/cursor upward or shift a region.
    # ACTION3: Move cursor right? Or change state of specific cells.
    # ACTION4: Move cursor left? Or same as ACTION1 but in opposite direction.
    # ACTIONs seem to move a "block" of colors (12, 9) and potentially update a status bar at bottom.
    
    new_grid = grid.copy()
    H, W = new_grid.shape
    
    if action == 1: # Upward movement / Shift
        # Looking at the deltas, ACTION1 shifts a vertical segment of color-coded blocks.
        # It seems to be moving a set of coordinates that represent 'active' elements.
        # launches a sequence of changes.
        # We need to find the current position of the blocks (color 12, 9) and move them up.
        # Find all pixels of color 12 or 9 that are not part of the large background structure.
        # 
        # In the initial grid, there are blocks of color 12 and 9.
        # Let's identify the block of interest.
        # The target block is usually located around x=19..23 or similar.
        # TheCores are likely the small clusters of 9s and 12s.
        # These blocks shift vertically by 5 units per ACTION1.
        shift = -5
        
        # Identify the "cursor" block: a cluster of colors 12, 9, 0, 1.
        # mask = (new_grid == 12) | (new_grid == 9) | (new_grid == 0) | (new_grid == 1)
        # But some 9s and 12s are in the same area as the 3s.
        # cursor_coords = np.argwhere(mask)
        
        # Based on observed transitions, ACTION1 moves specific rows.
        # It seems to be moving a vertical strip of width 5.
        # Find where the current 'active' block is.
        # We find the range of rows that have these special colors.
        # In the initial grid, they are at r45-r49.
        # In the first few ACTION1s, they move to r40-r44, then r35-r39, etc.
        # The target column range is usually x=19..23 or similar.
        # launches a sequence of changes.
        
        # Simple heuristic for this level: shift the active blocks up by 5.
        # For each column in the target range, we identify the pixels of color 12/9/0/1.
        #<|channel>thought
        # Let's try a more general approach based on the deltas.
        # ACTION1 shifts a region of height 5 and width 5.
        # ACTION4 shifts it horizontally? No, ACTION4 shifted from c19 to c24, then c29...
        # So ACTION4 = Move Right (by 5), ACTION3 = ?
        # Wait, ACTION3 moved from c24 back to c19. So ACTION3 = Move Left (by 5).
        # ACTION1 = Move Up (by 5).
        # ACTION2 = Move Down (by 5)? Not seen but likely.
        
        # Find the current "active block" center.
        # We look for colors 12, 9, 0, 1 that are not part of the static background.
        # The background is mostly color 4 and 3.
        # Special colors: {0, 1, 9, 12}
        mask = (new_grid == 12) | (new_grid == 9) | (new_grid == 0) | (new_grid == 1)
        coords = np.argwhere(mask)
        if coords.size == 0: return new_grid
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # The active block is a 5x5 area.
        # Let's find all pixels in this bounding box and move them.
        block_pixels = []
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if mask[y, x]:
                    block_pixels.append((y, x, new_grid[y, x]))
        
        # Move the block and restore the background.
        # Background for this level seems to be color 4 or 3.
        # We need to know what was there before. This is tricky without a full map.
        # But we can assume the background is whatever is at the destination if it's not special.
        
        # For ACTION1 (Up), shift y by -5.
        dy, dx = -5, 0
        if action == 1: dy, dx = -5, 0
        elif action == 2: dy, dx = 5, 0
        elif action == 3: dy, dx = 0, -5
        elif action == 4: dy, dx = 0, 5
        else: return new_grid
        
        # To avoid destroying other things, we first clear the old position.
        # The "background" is usually color 3 or 4.
        for y, x, val in block_pixels:
            # Determine background color based on coordinates
            bg = 3 if (8 <= y < 50 and 14 <= x < 54) else 4
            new_grid[y, x] = bg
            
        for y, x, val in block_pixels:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                new_grid[ny, nx] = val
        
        # Update status bar at bottom (r61-r62).
        # The status bar seems to track the cursor's X position.
        # It changes from c14 -> c15 -> c16... as ACTION1/ACTION4 are used.
        # Let's find the current marker in r61.
        status_row = 61
        marker_col = -1
        for x in range(W):
            if new_grid[status_row, x] == 3: # Marker is color 3
                marker_col = x
                break
        if marker_col != -1:
            # Shift marker based on action.
            # If ACTION1 moves it? The deltas show ACTION1 moving the marker too.
            # This suggests a complex relationship.
            pass

    return new_grid

def is_level_complete(grid):
    # Level complete when the active block reaches the top or a specific target.
    # In the win transition, ACTION1 was applied to a grid where the block was already high.
    # Looking at the "GRID BEFORE THE COMPLETING ACTION", the blocks of color 12/9 were around r15-r19.
    # Applying ACTION1 moved them further up and triggered completion.
    # A simple condition: if any special pixel (color 12) is above row 10.
    mask = (grid == 12) | (grid == 9)
    coords = np.argwhere(mask)
    if coords.size == 0: return False
    return coords[:, 0].min() < 10