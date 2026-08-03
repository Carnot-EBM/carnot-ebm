import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, this looks like a puzzle game where objects move in response to actions.
    # Action 4 seems to be moving an object (likely color 9) of size 1xN or Nx1 vertically/horizontally.
    # Action 1 moves something else.
    # Action 5 changes some specific cell values.
    # The rules are induced from the same provided delta sequences.
    
    new_grid = grid.copy()
    
    if action == 4: # Move Right
        # Find all cells of color 9 that are not part of a larger structure
        # We assume it's a "player" or "movable own object"
        # In these examples, the same set of rows (11-37) and columns (39-63) are changing.
        # It's as if a column of color 9 is moving right.
        # Shift a vertical bar of color 9 one pixel to the right.
        for r in range(11, 38):
            # find current position of color 9 in row r
            cols = np.where(grid[r] == 9)[0]
            if len(cols) > 0:
                # move each instance of 9 to the right by 1
                for c in cols:
                    new_grid[r, c] = 5 # restore original background
                    new_grid[r, min(c + 1, 63)] = 9
        # Special case for ACTION4 delta: r24 has complex changes.
        # Handle r24 specifically as it'<|channel>thought
        # The observed deltas show r24 shifting its content.
        # For simplicity, we implement a general shift.
    
    elif action == 1: # Move Up
        # Similar logic to Action 4 but vertically.
        # Find all cells of color 9 or other movable objects and shift them up.
        # Find rows where color 9 exists.
        rows = np.where(grid == 9)[0]
        if len(rows) > 0:
            # Find the bounding box of the "player" object (color 9).
            # la = grid[row, col] = 9
            pass

    # To match the provided transitions exactly, we would need to actually track the state.
    # as if there is a player at some position.
    # a//C-A-T-S
    # This is likely a puzzle game with specific movement rules.
    # Since I must provide an executable world model, I will implement the most plausible simple rule.
    
    # In this specific dataset, ACTION 4 shifts something right, ACTION 1 shifts something up.
    # Let's refine based on the delta patterns.
    # ACTION 4: r11c39->r11c42, then r11c42->r11c45... it's moving a vertical bar of color 9 from c=39 to c=42 to c=45.
    # ACTION 1: moves things vertically.
    
    # We can see that in action 4, cells are changing from 9 to 5 and 5 to 9.
    # The column index increases by 3 each time.
    
    # For Action 4 (Right):
    if action == 4:
        for r in range(64):
            row = grid[r]
            # Find where 9s are and shift them right by 3
            mask_9 = (row == 9)
            new_grid[r][mask_9] = 5
            # Shift mask_9 right by 3
            shifted_mask = np.roll(mask_9, 3)
            # Correct for wrap-around
            shifted_mask[:3] = False
            new_grid[r][shifted_mask] = 9
            
    elif action == 1: # Up
        # Based on the deltas, this is shifting something up.
        # Let's implement a simple shift up by 3 rows.
        mask_9 = (grid == 9)
        new_grid[mask_9] = 5
        shifted_mask = np.roll(mask_9, -3, axis=0)
        shifted_mask[-3:] = False
        new_grid[shifted_mask] = 9

    return new_grid

def is_level_complete(grid):
    # The win state isn't explicitly given as a full grid, but usually it involves
    # reaching a target or clearing objects. In many ARC games, color 0 or specific patterns are goals.
    # Looking at r63c58:1x1 in ACTION4, and others, there might be a counter.
    # For now, return False unless a clear condition is met.
    return False