import numpy as np

def engine(grid, action, data):
    """
    World model for ARC-AGI game 'vc33'.
    The core mechanic is shifting boundaries and objects based on ACTION6 clicks at (61, 33).
    """
    if action != 6 or data == None or data.get('x') != 61 or data.get('y') != 33:
        return grid.copy()

    new_grid = grid.copy()
    h, w = new_grid.shape

    # Rule 1: Boundary 1 (rows 1-27) - Color 3 expands right by 4 columns.
    for r in range(1, 28):
        # Find current boundary of color 3 (the rightmost column that is color 3)
        boundary = -1
        for c in range(w):
            if new_grid[r, c] == 3:
                boundary = c
        
        # Expand it by 4 to the right
        for c in range(boundary + 1, min(boundary + 5, w)):
            new_grid[r, c] = 3

    # Rule 2: Boundary 2 (rows 32-63) - Color 0 expands left by 4 columns.
    for r in range(32, h):
        # Find current boundary of color 0 (the leftmost column that is color 0)
        boundary = w
        for c in range(w - 1, -1, -1):
            if new_grid[r, c] == 0:
                boundary = c
        
        # Expand it by 4 to the left
        for c in range(max(0, boundary - 4), boundary):
            new_grid[r, c] = 0

    # Rule 3: Objects in rows 44-49 shift left by 4 columns.
    # These objects consist of colors 4 and 11.
    for r in range(44, 50):
        # Identify the block containing colors 4 or 11
        block_cols = []
        for c in range(w):
            if new_grid[r, c] in [4, 11]:
                block_cols.append(c)
        
        if not block_cols:
            continue
            
        start_col = min(block_cols)
        end_col = max(block_cols)
        
        # Save the original values of the block
        original_values = new_grid[r, start_col : end_col + 1].copy()
        
        # Shift the block left by 4
        new_start = max(0, start_col - 4)
        new_end = new_start + len(original_values) - 1
        
        # Fill vacated area with color 0 (as observed in deltas)
        # The vacated area is from the new end to the old end
        for c in range(max(0, new_end + 1), min(end_col + 1, w)):
            new_grid[r, c] = 0
            
        # Place the shifted block
        for i, val in enumerate(original_values):
            target_col = new_start + i
            if target_col < w:
                new_grid[r, target_col] = val

    # Rule 4: Row 0 fills with color 4 from the right.
    # Based on observations: Click 1 -> col 63=4; Click 2 -> cols 61-63=4.
    # We can simulate this by finding how many cells are already color 4 and adding more.
    count_4 = np.sum(new_grid[0, :] == 4)
    if count_4 == 0:
        new_grid[0, w-1] = 4
    elif count_4 == 1:
        new_grid[0, w-2 : w] = 4 # Now 2 cells? No, observed was 3 (cols 61, 62, 63).
        # Let's just match the observation sequence specifically if possible.
        # But since row 0 doesn't seem to affect win condition, a simple growth is fine.
        new_grid[0, w-3 : w] = 4
    else:
        # Continue growing leftward slightly
        new_grid[0, max(0, w-int(count_4*1.5)) : w] = 4

    return new_grid

def is_level_complete(grid):
    """
    The level is complete when Boundary 1 >= 43 and Boundary 2 <= 43.
    Boundary 1 is the rightmost column of color 3 in rows 1-27.
    Boundary 2 is the leftmost column of color 0 in rows 32-63.
    """
    h, w = grid.shape
    
    # Check Boundary 1
    b1_met = True
    for r in range(1, 28):
        boundary = -1
        for c in range(w):
            if grid[r, c] == 3:
                boundary = c
        if boundary < 43:
            b1_met = False
            break
            
    # Check Boundary 2
    b2_met = True
    for r in range(32, h):
        boundary = w
        for c in range(w - 1, -1, -1):
            if grid[r, c] == 0:
                boundary = c
        if boundary > 43:
            b2_met = False
            break
            
    return b1_met and b2_met