import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION1: Up
    # ACTION2: Down
    # ACTION3: Left
    # ACTION4: Right
    # ACTION5: ?
    # ACTION6: Click
    # ACTION7: ?
    
    # The game seems to be a puzzle where an object (a "player" or "cursor") moves within a boundary.
    # Based on the same-shape tracking and the delta changes, it looks like there's a//
    # a moving region/object that interacts with the same-color cells.
    # Let's identify the "active" object. In the initial grid, color 3 is a large rectangle.
    # laout:
    # Color 4: Background
    # Color 3: Boundary / Wall
    # Color 5: Player/Active area?
    # Color 9: Obstacles/Items
    # Color 0, 1: Special markers
    # Color 11: Goal area?
    # Color 8: Win condition targets?
    
    # Looking at the deltas:
    # ACTION3 (Left): r45c24 -> r45c19... shifts the active block of color 5/9/12 left.
    # ACTION4 (Right): r25c19 -> r25c24... shifts the active block right.
    # ACTION1 (Up): r40c19 -> r35c19... shifts the active block up.
    # ACTION2 (Down): Not observed but inferred as Down.
    # ACTION3 (Left): Shifts x by -5.
    # ACTION4 (Right): Shifts x by +5.
    # ACTION1 (Up): Shifts y by -5.
    # ACTION2 (Down): Shifts y by +5.
    
    # The "active" object is a rectangle of size 5x5 or similar that moves in increments of 5.
    # Let's find the current position of this moving object.
    # In the initial grid, there's a region of color 5 and others inside the boundary (color 3).
    # Inside the boundary (color 3), we identify the non-background (non-4) cells.
    # Find all coordinates of cells with colors other than background (4) and boundary (3).
    # This is a---
    #
    # Actually, looking at the<|channel>thought process:
    # The movement is exactly 5 units.
    # The target area seems to be the bottom row (r61, r62).
    # When the player moves, some values in r61/r62 change.
    # These changes are like a cursor moving across the goal line.
    #
    # The active block is located between rows 8 and 49, columns 14 and 53.
    # It moves within these bounds.
    #
    # Let's refine the engine:
    # 1. Identify the "active" block: the cluster of cells that aren't 3 or 4.
    # 2. Move it by 5 pixels in the given direction.
    # 3. Update the grid.
    # 4. Update the "cursor" on the bottom row based on the center of the active block.
    
    new_grid = grid.copy()
    
    # Define boundaries for the active object
    y0, x0, y1, x1 = 8, 14, 49, 53
    
    # Find current position of the active object (the non-3, non-4 cells)
    coords = np.argwhere((grid != 3) & (grid != 4))
    if len(coords) == 0:
        return new_grid
    
    # We only care about the ones inside the boundary [8, 49]x[14, 53]
    mask = (coords[:, 0] >= y0) & (coords[:, 0] <= y1) & (coords[:, 1] >= x0) & (coords[:, 1] <= x1)
    active_coords = coords[mask]
    
    if len(active_coords) == 0:
        return new_grid

    # Calculate movement
    dy, dx = 0, 0
    if action == 1: dy = -5
    elif action == 2: dy = 5
    elif action == 3: dx = -5
    elif action == 4: dx = 5
    
    # Shift coordinates
    new_coords = active_coords + [dy, dx]
    
    # Check if move is valid (stays within boundaries)
    # The object has a certain width/height. Let's check all points.
    if np.any((new_coords[:, 0] < y0) | (new_coords[:, 0] > y1) | 
              (new_coords[:, 1] < x0) | (new_coords[:, 1] > x1)):
        return grid # Invalid move
    
    # Update the grid: clear old position and set new position
    for r, c in active_coords:
        new_grid[r, c] = 4 # Reset to background
    for r, c in new_coords:
        # We need to preserve the colors of the moving object
        # Map old coord to new coord color
        # Since it's a simple shift, we can just use the original grid values
        # But wait, the object might overlap with itself or other things.
        # A better way: create a temporary copy of the object's pixels.
        pass

    # Correct implementation of shifting the object
    object_pixels = {}
    for r, c in active_coords:
        object_pixels[(r, c)] = grid[r, c]
    
    for r, c in active_coords:
        new_grid[r, c] = 4
        
    for (r, c), val in object_pixels.items():
        new_grid[r + dy, c + dx] = val
        
    # Now update the cursor at the bottom (rows 61, 62)
    # The observed deltas show changes at r61c14, r61c15... as the block moves.
    # This looks like the x-coordinate of the block is mapped to the column index.
    # Initial center X was around 36? Let's see.
    # ACTION3 (Left): shifted from c24 to c19. Cursor moved from c14 to c15? No.
    # Let's look closer: "r61c14:3x1" then "r61c15:3x1".
    # It seems the cursor position is simply related to the current X offset.
    # If we move left (-5), the cursor might move right (+1)? Or vice versa.
    # Actually, looking at "ACTION3 (level 0->0): changed cells ... r61c14:3x1", 
    # it means cell (61, 14) became color 3.
    # Then "ACTION3 (level 0->0): changed cells ... r61c15:3x1", 
    # it means cell (61, 15) became color 3.
    # So moving Left (dx=-5) increments the cursor column by 1.
    # Moving Right (dx=+5) decrements the cursor column by 1.
    # Wait, that's counter-intuitive. Let me re-read.
    # ACTION3 (Left) -> r61c14:3x1, then next ACTION3 (Left) -> r61c15:3x1.
    # Yes, each ACTION3 moves the cursor one step to the right.
    # Each ACTION4 (Right) moves the cursor one step to the left.
    # This is a mirrored mapping.
    
    # To implement this precisely, we need to track the current cursor position.
    # The cursor consists of color 3 pixels in rows 61 and 62.
    cursor_cols = np.argwhere((grid[61] == 3))[:, 0]
    if len(cursor_cols) > 0:
        curr_col = cursor_cols[0]
        new_col = curr_col + (1 if action == 3 else -1 if action == 4 else 0)
        if 14 <= new_col < 54:
            new_grid[61, curr_col] = 11 # Reset old cursor to goal color
            new_grid[62, curr_col] = 11
            new_grid[61, new_col] = 3  # Set new cursor
            new_grid[62, new_col] = 3
            
    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a target or clearing objects.
    # In this game, it might be when the cursor reaches the far right/left
    # or when the active block reaches a certain position.
    # Given no win state grid, we'll assume it's complete when the cursor 
    # hits one of the targets (color 8) at the end of row 61.
    # Targets are at c56, c59, c62.
    cursor_cols = np.argwhere((grid[61] == 3))[:, 0]
    if len(cursor_cols) > 0:
        col = cursor_cols[0]
        if col in [56, 59, 62]:
            return True
    return False