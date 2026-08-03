import numpy as np

def engine(grid, action, data):
    """
    The game involves clicking on specific regions to trigger changes.
    Based on the observed transitions:
    - Clicking at y=59 seems to clear/set a region in the bottom area (r56-r61).
    - Clicking at y=30 triggers two things: it fills a rectangle in the middle 
      area (r28-r31) and resets/fills a corresponding region in the bottom area.
    - The colors being filled are associated with the x-coordinate of the click.
    - Specifically, clicks at y=30 fill rectangles of color 9, 14, or 11 based on x.
    - There's also a small change at r53c63, r53c62, etc., which looks like a counter.
    """
    if action != 6:
        return grid.copy()

    new_grid = grid.copy()
    px, py = data['x'], data['y']

    # Logic for clicking at y=59 (Bottom Area Clear/Reset)
    if py == 59:
        # Mapping px to column ranges in the bottom section
        # Observed: x=36 -> c33:c38, x=20 -> c17:c22, x=44 -> c41:c46
        col_start = px - 3 # Approximate offset from observations
        for r in range(56, 62):
            if r == 56 or r == 61:
                new_grid[r, col_start : col_start + 6] = 0
            elif 57 <= r <= 60:
                new_grid[r, col_start] = 0
                new_grid[r, col_start + 5] = 0
        return new_grid

    # Logic for clicking at y=30 (Middle Area Fill and Bottom Reset)
    if py == 30:
        # Determine color based on x coordinate
        # x=23 -> color 9, x=29 -> color 14, x=35 -> color 11
        color = 0
        if px == 23: color = 9
        elif px == 29: color = 14
        elif px == 35: color = 11
        
        # Middle area fill (r28-r31)
        col_start = px - 2
        for r in range(28, 32):
            new_grid[r, col_start : col_start + 4] = color
            
        # Update counter at row 53
        if px == 23: new_grid[53, 63] = 3
        elif px == 29: new_grid[53, 62] = 3
        elif px == 35: new_grid[53, 61] = 3

        # Corresponding bottom area reset/fill
        # Mapping px to column ranges for the bottom section
        bottom_col_start = {23: 33, 29: 17, 35: 41}
        bc = bottom_col_start.get(px, 0)
        for r in range(56, 62):
            if r == 56 or r == 61:
                new_grid[r, bc : bc + 6] = 4
            elif 57 <= r <= 60:
                # Special pattern observed: 4x2, 2x2, 4x2 (total 6 wide)
                # This implies indices [bc, bc+1]=4, [bc+2, bc+3]=2, [bc+4, bc+5]=4
                new_grid[r, bc : bc + 2] = 4
                new_grid[r, bc + 2 : bc + 4] = 2
                new_grid[r, bc + 4 : bc + 6] = 4
        return new_grid

    return new_grid

def is_level_complete(grid):
    """
    The win state isn't explicitly provided, but typically it involves 
    filling specific targets or clearing the board. Given the data, 
    we return False as no clear win condition was demonstrated.
    """
    return False