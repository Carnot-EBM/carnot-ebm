import numpy as np


def engine(grid, action, data):
    grid = grid.copy()
    if action == 6 and data is not None:
        px, py = int(data['x']), int(data['y'])
        # Determine which "slot" was clicked (left or right) based on x coordinate
        slot = 0 if px < 32 else 1
        base_col = 62 - slot * 4
        # Check if there's a 15-block at the click location
        if grid[py, px] == 15:
            # Find all 15-blocks in the same column region
            cols_to_check = range(base_col, base_col + 4)
            for c in cols_to_check:
                for r in range(64):
                    if grid[r, c] == 15:
                        grid[r, c] = 5
                        break
        # Move the score indicator left by 2 cells
        # The score bar is at row 63, starting from col 62 going left
        # Find current rightmost non-5 cell in row 63
        # Actually, let's track it differently: find where the 5s end in row 63
        # Row 63 starts as all 0s. Each action adds 2 cells of color 5 to the right side.
        # We need to find the current extent and extend it.
        
        # Find the leftmost 5 in row 63 (the score bar extends from right)
        row63 = grid[63, :]
        # Find rightmost position that is already 5
        # The bar grows from right to left
        # Find the boundary: scan from right, find first non-5
        pos = 63
        while pos >= 0 and row63[pos] == 5:
            pos -= 1
        # pos is now the last non-5 cell; the next two cells to the right are new
        # But wait - the bar might not be contiguous yet if we're just starting
        # Let me re-examine: initially row 63 is all 0s.
        # After first click: r63c62:5x2 means cols 62,63 become 5
        # After second: r63c60:5x2 means cols 60,61 become 5
        # So each action sets 2 more cells to 5, moving leftward
        
        # Find where the existing bar starts (leftmost 5)
        leftmost_5 = None
        for c in range(64):
            if row63[c] == 5:
                leftmost_5 = c
                break
        
        if leftmost_5 is None:
            # No bar yet, start at col 62
            grid[63, 62] = 5
            grid[63, 63] = 5
        else:
            # Extend 2 cells to the left of current leftmost
            new_start = max(leftmost_5 - 2, 0)
            for c in range(new_start, leftmost_5):
                grid[63, c] = 5
    
    return grid


def is_level_complete(grid):
    # Check if all 9-colored cells have been converted (collected)
    # The win condition appears to be when no color-9 cells remain
    return not np.any(grid == 9)