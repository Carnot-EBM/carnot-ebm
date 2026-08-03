import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # Action 6 is a click at (px, py). Logical coords are px, py.
    # Based on the observed transitions, clicking a cell changes its neighborhood.
    # The game seems to be about moving a 3x3 block of color 15 and interacting with other elements.
    # In this case, the same coordinates provided in the<|channel>thought process were actually used.
    # x=10, y=53 -> r52c9, r53c9, r54c9 (a 3x3 block) becomes 15s.
    # Also some cells change from 15 to 5.
    # The "block" moves to the clicked location.
    # The interaction happens when it lands on specific colors.
    # The "target" area is likely the bottom row or similar.
    # The target pixels are probably related to the<|channel>thought process.
    # The laout has several markers (color 3) and targets (color 15 initially).
    # Let's implement the movement of the 3x3 block of color 15.
    
    grid_next = grid.copy()
    
    # Find current position of the 3x3 block of color 15
    # We look for the top-left corner of the 3x3 block.
    rows, cols = np.where(grid == 15)
    if len(rows) == 0:
        return grid_next
    
    # Assuming there is always exactly one 3x3 block of color 15
    min_r, min_c = np.min(rows), np.min(cols)
    
    # Clear old block
    grid_next[min_r:min_r+3, min_c:min_c+3] = 5 # Use background color of that region
    # Note: This is a bit simplistic; we need to determine the background color.
    # In the observed transitions, it seems the background changes based on where it was.
    # Background colors are 4 or 5.
    bg_color = grid[min_r, min_c] if grid[min_r, min_c] == 15 else 5
    # Actually, let's just use the value from the delta which shows it becomes 5 in most cases.
    # bg_color = 5
    
    # New position (px, py) provided in data['x'], data['y']
    # The click coordinates are px, py.
    # The 3x3 block top-left corner should be around (py-1, px-1).
    # Target top-left: r=py-1, c=px-1
    new_r, new_c = data['y'] - 1, data['x'] - 1
    
    # Place new block
    grid_next[new_r:new_r+3, new_c:new_c+3] = 15
    
    # Interaction with markers (color 3):
    # When a block moves to a location, it "collects" marker 3.
    # When it collects a marker, something happens at the bottom row (r63).
    # In the observed transitions, r63 changes from 0 to 5.
    # Let's check if the new block covers any cell of color 3.
    collected = False
    for r in range(new_r, new_r + 3):
        for c in range(new_c, new_c + 3):
            if grid[r, c] == 3:
                collected = True
                break
        if collected: break
    
    if collected:
        # Find current progress on the bottom row (r63)
        # The delta shows r63c62:5x2, then r63c60:5x2, etc.
        # It fills from right to left.
        # Fill two cells of color 5 starting from the rightmost available.
        # Current filled length is sum of cells that are not 0 in r63.
        filled_len = np.sum(grid[63, :] != 0)
        # We fill 2 more cells of color 5.
        # The width is 64.
        # Start filling from index 63 - filled_len - 1? No, let's look at the data.
        # First action: r63c62:5x2 -> indices 62, 63 are now 5.
        # Second action: r63c60:5x2 -> indices 60, 61 are now 5.
        # Third action: r63c58:5x2 -> indices 58, 59 are now 5.
        # This means every time a marker is collected, we fill 2 pixels on row 63 from right to left.
        start_col = 64 - (filled_len + 2)
        if start_col >= 0:
            grid_next[63, start_col : start_col + 2] = 5
    
    return grid_next

def is_level_complete(grid):
    # Level complete when all markers (color 3) are gone or bottom row is full.
    # Based on the pattern, it's likely when the bottom row is fully filled with color 5.
    return np.all(grid[63, :] == 5)