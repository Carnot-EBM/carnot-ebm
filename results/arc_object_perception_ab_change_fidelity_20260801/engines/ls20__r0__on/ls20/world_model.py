import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Actions: ACTION1=Up, ACTION2=Down, ACTION3=Left, ACTION4=Right, ACTION6=Click
    # The game seems to be a puzzle where an object (a "player" or "cursor") moves within a boundary.
    # Based on the observed transitions, it's a moving region/object that changes colors of cells.
    # Let's identify the player object. In the INITIAL GRID, there's a lot of color 4 and 3.
    # Color 3 is a large rectangular area.
    # Looking at the delta for ACTION3 (Left), r45c24:12x5,3x5 etc. means columns 24-28 are now 12 and 29-33 are 3.
    # It looks like a window of size 5x5 (or similar) is sliding across the same set of rows.
    # Rows 40-49 are involved in some movements.
    # Actually, looking at the ACTION1 (Up) deltas, the row indices decrease.
    #//
    # Let's refine the model:
    # There is a "brush" or "window" of width 5 and height 10 (rows 40-49).
    # When action is 1 (Up), the brush moves up by 5 rows.
    # When action is 3 (Left), the brush moves left by 5 cols.
    # When action is 4 (Right), the brush moves right by 5 cols.
    # The brush seems to be replacing colors based on some pattern.
    # But wait, the laest transitions show it's moving over color 3 area.
    # Color 3 is the background for the puzzle part.
    # Looking at the ACTION1 (Up) deltas: r40c19:12x5... r35c19:12x5... r30c19:12x5... r25c19:12x5.
    # It looks like the brush is a 10x5 block that changes cells to a specific color (e.g., 12 or 9).
    # The same block is moveing across the grid.
    # Let's track the position of the player/brush.
    # In the INITIAL GRID, there's an object of color 12 and 9 in rows 45-49.
    # Specifically, r45c29:12x5, r46c29:12x5, r47c29:9x5, r48c29:9x5, r49c29:9x5.
    # This is a 5x5 block of color 12 (top half) and 9 (bottom half).
    # Brush size: height=5, width=5.
    # Initial pos: top_row = 45, left_col = 29.
    # Movement: ACTION1(Up): row -= 5; ACTION2(Down): row += 5; ACTION3(Left): col -= 5; ACTION4(Right): col += 5.
    # The brush replaces colors of cells it covers.
    # But wait, the deltas show that when it moves, the old position becomes color 3 again?
    # Yes: "r45c24:12x5,3x5" means at col 24, we have 5 pixels of 12, then 5 pixels of 3.
    # So the brush is moving and leaving behind color 3.
    # Let's find the current brush position by looking for color 12 or 9 in the grid.
    # la//
    # Let's try to implement this.

    new_grid = grid.copy()
    
    # Find the brush: a block of color 12 or 9.
    # We search for the first occurrence of color 12.
    brush_pos = None
    for r in range(64):
        for c in range(64):
            if grid[r, c] == 12:
                brush_pos = (r, c)
                break
        if brush_pos: break
    
    if brush_pos is None:
        return new_grid
    
    curr_row, curr_col = brush_pos
    
    # The brush is actually a 5x5 block.
    # Top half (rows 0-2) are color 12, and bottom half (rows 3-4) are color 9?
    # No, let's look at INITIAL GRID again: r45c29:12x5, r46c29:12x5, r47c29:9x5...
    # So rows 0-1 are 12, rows 2-4 are 9. Wait, that's 5 rows total.
    # Let's check ACTION1 (Up) delta: r40c19:12x5, r41c19:12x5, r42c19:9x5...
    # This is exactly the same pattern: 2 rows of 12, 3 rows of 9.
    # Brush height = 5, width = 5.
    
    # Movement
    dr, dc = 0, 0
    if action == 1: dr = -5
    elif action == 2: dr = 5
    elif action == 3: dc = -5
    elif action == 4: dc = 5
    
    new_row, new_col = curr_row + dr, curr_col + dc
    
    # Boundary checks (simplified, as we don't see boundaries in deltas)
    # The brush moves within a region.
    # Color 3 area is the "track".
    # Let's assume it can move anywhere but doesn't go off grid.
    new_row = max(0, min(63, new_row))
    new_col = max(0, min(63, new_col))
    
    # Restore old position to color 3 (or whatever was there before).
    # In the observed transitions, the cells are restored to color 3.
    for r in range(curr_row, curr_row + 5):
        for c in range(curr_col, curr_col + 5):
            if r < 64 and c < 64:
                new_grid[r, c] = 3
    
    # Draw new position
    for r in range(new_row, new_row + 5):
        for c in range(new_col, new_col + 5):
            if r < 64 and c < 64:
                # Pattern: first 2 rows of 12, last 3 rows of 9.
                color = 12 if (r - new_row) < 2 else 9
                new_grid[r, c] = color
                
    return new_grid

def is_level_complete(grid):
    # The win state is not provided, but usually it's when a certain object reaches a target.
    # Let's assume it's complete when the brush is at some specific location or no longer exists.
    # We search for any cell of color 12 or 9 that is NOT part of the same block.
    # Or maybe it's just based on the brush reaching a certain coordinate.
    # For now, return False as we don't see a WIN STATE grid.
    return False