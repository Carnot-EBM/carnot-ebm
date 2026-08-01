import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # Action 6 is a click at (px, py). Logical coordinates are same as pixel coords.
    px, py = data['x'], data['y']
    
    # The goal seems to be moving a 3x3 block of color 15 (the "player" or "cursor")
    # and potentially interacting with other elements.
    # Based on the transitions, clicking a point (x, y) moves the 3x3 block of color 15 to center it around that point?
    # No, looking at the delta: ACTION6 x=10, y=53 -> r52c9:15x3, r53c9:15x3, r54c9:15x3.
    # This means the top-left corner of the 3x3 block is (py-1, px-1).
    # Or more than simply, if we're clicking at (px, py), the 3x3 block is placed at rows [py-1, py+1] and cols [px-1, px+1].
    
    # Find current position of the 3x3 block of color 15.
    # We need to find where color 15 exists first.
    rows, cols = np.where(grid == 15)
    if len(rows) == 0:
        # If no color 15 block exists, we use the same initial state provided in the initial grid.
        # start_pos = (4, 30) # Initial block pos from INITIAL GRID
        return grid.copy()

    # Save existing block coordinates for reference if __name__ == "__main__":
    # a new 3x3 block will bes set to 15s.
    # curr_////////////// ////////////////// own positions
    #<|channel>thought
    # The observed transitions show that when ACTION6 (click) occurs, a 3x3 block of color 15 is moved to center it around (px, py).
    # However, the cells it leaves behind are restored to their "background" colors.
    # Based on the data, r58c3:5x3, r59c3:5x3, r60c3:5x3 were changed back to color 5.
    # This implies there was a 3x3 block at (58, 3) and it moved to (52, 9).
    # Wait, let's check the first transition: click x=10, y=53 -> new block at r52-54, c9-11.
    # Previous block was at r58-60, c3-5? No, INITIAL GRID says r58-60, c3-5 is color 15.
    # So clicking (10, 53) moves the block from (58, 3) to (52, 9).
    # The background colors are 4 or 5.
    # Let's assume the grid has a static "underlay" or we just restore based on common patterns.
    # Most of the board is 4 or 5.
    # Rows 0-9 are mostly 4. Rows 10-62 are mostly 5. Row 63 is 0.
    
    new_grid = grid.copy()
    
    # Find all current cells of color 15
    rows_15, cols_15 = np.where(grid == 15)
    for r, c in zip(rows_15, cols_15):
        # Restore cell to its likely background: 4 if r < 10 else 5
        new_grid[r, c] = 4 if r < 10 else 5
        if r == 63: new_grid[r, c] = 0

    # Place new 3x3 block centered at (px, py)
    # Note: data['x'] is px, data['y'] is py.
    # Transition x=10, y=53 -> rows [52, 53, 54], cols [9, 10, 11].
    # This means top-left is (py-1, px-1).
    tr, tc = py - 1, px - 1
    for r in range(tr, tr + 3):
        for c in range(tc, tc + 3):
            if 0 <= r < 64 and 0 <= c < 64:
                new_grid[r, c] = 15
                
    return new_grid

def is_level_complete(grid):
    # No win state grid provided, but usually it's when the player reaches a target or clears something.
    # In this case, we don't have enough info to determine the win condition.
    # Let's assume it's not complete unless specified.
    return False