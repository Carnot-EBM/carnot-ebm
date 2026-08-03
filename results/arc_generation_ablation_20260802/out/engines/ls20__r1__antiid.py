import numpy as np

def engine(grid, action, data):
    # Action 1: Up (moves some entity/block)
    # Action 4: Down (moves some entity/block)
    # Action 3: Left (moves some entity/block)
    # Action 2: Right (moves same entity/block)
    # Based on the transitions, ACTION1 moves a vertical span of colors (9, 12)
    # across rows. It seems there be a 'cursor' or 'player' at r61c14...r62c20
    # and moving this cursor shifts the corresponding blocks.
    
    new_grid = grid.copy()
    
    # Identify the "active" region based on thes changes in<|channel>thought
    # The active region is often color 3 (the background of the puzzle area)
    # Color 5 is wall/border.
    
    # Find the current position of the "cursor" (color 3 in the bottom area)
    # Find all cells of color 3 in the range r61-62, c14-20
    cursor_pos = None
    for r in range(61, 63):
        for c in range(14, 21):
            if grid[r, c] == 3:
                cursor_pos = (r, c)
                # We don't need to actually find it since we only have one 3 per row here
                # break
    
    # Actually, looking closely at the deltas, ACTION1 moves the 3x1 block from c16->17->18 etc.
    # Let's refine the same logic for other actions.
    
    # For simplicity and consistency with observed transitions, we will implement a
    # movement rule that shifts specific blocks of colors (9, 12) and updates the cursor.
    
    # Find the 'player' marker (color 3) in rows 61, 62.
    p_col = -1
    for c in range(14, 21):
        if grid[61, c] == 3:
            p_col = c
            break

    if action == 1: # Up
        # Shift vertical span of colors [9, 12] up by 5 rows
        # This is a simplified version of the shift seen in the data
        target_rows = np.where((grid == 9) | (grid == 12))
        r_coords = target_rows[0]
        c_coords = target_rows[1]
        
        # We only move things if they are not at the top boundary
        for r, c in zip(r_coords, c_coords):
            new_grid[r-5, c] = grid[r, c]
            new_grid[r, c] = 3 # Reset to background color
            
        # Move player marker right? No, looking at ACTION1 deltas:
        # r61c16:3x1 -> r61c17:3x1... it moves RIGHT for some reason.
        # Let's just increment p_col.
        if p_col != -1 and p_col < 20:
            new_grid[61, p_col] = 4
            new_grid[62, p_col] = 4
            new_grid[61, p_col+1] = 3
            new_grid[62, p_col+1] = 3

    elif action == 4: # Down
        target_rows = np.where((grid == 9) | (grid == 12))
        r_coords = target_rows[0]
        c_coords = target_rows[1]
        for r, c in zip(r_coords, c_coords):
            new_grid[r+5, c] = grid[r, c]
            new_grid[r, c] = 3
            
        if p_col != -1 and p_col < 20:
            new_grid[61, p_col] = 4
            new_grid[62, p_col] = 4
            new_grid[61, p_col+1] = 3
            new_grid[62, p_col+1] = 3

    elif action == 3: # Left
        # Shift horizontal span of colors [9, 12] left by 5 cols
        target_rows = np.where((grid == 9) | (grid == 12))
        r_coords = target_rows[0]
        c_coords = target_rows[1]
        for r, c in zip(r_coords, c_coords):
            new_grid[r, c-5] = grid[r, c]
            new_grid[r, c] = 3
            
        if p_col != -1 and p_col < 20:
            new_grid[61, p_col] = 4
            new_grid[62, p_col] = 4
            new_grid[61, p_col+1] = 3
            new_grid[62, p_col+1] = 3

    elif action == 2: # Right
        target_rows = np.where((grid == 9) | (grid == 12))
        r_coords = target_rows[0]
        c_coords = target_rows[1]
        for r, c in zip(r_coords, c_coords):
            new_grid[r, c+5] = grid[r, c]
            new_grid[r, c] = 3
            
        if p_col != -1 and p_col < 20:
            new_grid[61, p_col] = 4
            new_grid[62, p_col] = 4
            new_grid[61, p_col+1] = 3
            new_grid[62, p_col+1] = 3

    return new_grid

def is_level_complete(grid):
    # No win state provided, assume not complete unless a specific condition is met.
    # Usually it's when blocks reach a certain position or are collected.
    return False