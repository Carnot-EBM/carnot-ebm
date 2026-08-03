import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action 6 is a click at (data['x'], data['y'])
    # Actions 1-5 are keyboard/directional movements or triggers.
    
    # Based on observed transitions, ACTION 6 seems to be a "clear" or "fill" operation 
    # that replaces a region of cells with color 10.
    # ACTION 3 and ACTION 4 seem to be "toggle" or "shift" operations that change colors 
    # in specific patterns.
    # The game state contains complex repeating patterns of colors [0, 3, 5, 7, 9, 10, 11, 14, 15].
    # The same cell values are<|channel>thoughtly changing based on actions.
    # The laout suggests a puzzle where certain regions are filled or cleared.
    
    new_grid = grid.copy()
    
    if action == 6:
        # Click at (px, py)
        # In the observations, clicking at (24, 36) changed r37c25:10x6...r41c25:10x6
        # This implies a block of size 5x6 starting at row 37, col 25.
        # px=24, py=36 -> row=py+1, col=px+1? No, let's look closer.
        # data={'x': 24, 'y': 36} -> r37-41, c25-30.
        # Row range: y+1 to y+5. Col range: x+1 to x+6.
        # Let's generalize this as filling a 5x6 area with color 10.
        x, y = data['x'], data['y']
        new_grid[y+1 : y+6, x+1 : x+7] = 10
        
    elif action == 3:
        # ACTION 3 seems to be a "pattern shift" or "restore".
        # It replaces blocks of 10s with specific patterns involving colors 5, 9, 11.
        # In the first transition, it changed r37c37:5x2, 9x1, 5x2... which is length 5.
        # The pattern for rows 37 and 40 is [5, 5, 9, 5, 5].
        # The pattern for rows 38 and 39 is [5, 11, 9, 9, 5].
        # This happens in areas where there were 10s.
        # We need to find regions of 10s and replace them with these patterns.
        # However, since we must be deterministic and simple, let's look at the delta.
        # The deltas are very specific. Let's try to implement a simplified version.
        
        # Find all 5x5 or 5x6 blocks of color 10 and replace them.
        # For simplicity, if we see a block of 10s, we apply the observed pattern.
        for r in range(grid.shape[0] - 4):
            for c in range(grid.shape[1] - 4):
                if np.all(grid[r:r+5, c:c+5] == 10):
                    # Apply pattern
                    new_grid[r, c:c+5] = [5, 5, 9, 5, 5]
                    new_grid[r+1, c:c+5] = [5, 11, 9, 9, 5]
                    new_grid[r+2, c:c+5] = [5, 11, 9, 9, 5]
                    new_grid[r+3, c:c+5] = [5, 5, 9, 5, 5]
                    new_grid[r+4, c:c+5] = [5, 5, 5, 10, 10] # Simplified based on r41c38:10x3
                    break # Only one block per action for simplicity
                    
    elif action == 4:
        # ACTION 4 seems to be a "shift" or "partial clear".
        # It changes some patterns back to 10s or shifts them.
        for r in range(grid.shape[0] - 4):
            for c in range(grid.shape[1] - 4):
                if np.all(grid[r:r+5, c:c+5] != 10):
                    # This is too vague. Let's just return the grid if we can't induce it.
                    pass

    # The cell at r63c... also increments its value (or moves) with every action.
    # This looks like a progress bar or counter.
    # Find the first non-zero/non-background cell in row 63 and move it right.
    row_63 = new_grid[63].copy()
    nonzero_idx = np.where(row_63 != 0)[0]
    if len(nonzero_idx) > 0:
        idx = nonzero_idx[0]
        new_grid[63, idx] = 0
        if idx + 1 < new_grid.shape[1]:
            new_grid[63, idx + 1] = 15 # Color from observed delta r63c5:15x1
            
    return new_grid

def is_level_complete(grid):
    # Win state usually involves clearing all target blocks or reaching the end of the progress bar.
    # In this game, let's assume completion when the marker in row 63 reaches the far right.
    return np.any(grid[63, -1:] == 15)