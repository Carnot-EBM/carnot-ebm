import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape)
    # Action 3: Move left/right? No, looking at ACTION3 transitions, it seems like a "shift" or "toggle" operation.
    # Let's analyze ACTION3 more closely. It looks like it replaces a 5x5 area with a pattern.
    # The pattern in ACTION3 is r37c37:5x2,9x1,5x2 etc.
    # In the observed transitions, ACTION3 shifts the same pattern leftwards by 6 columns each time.
    # Also, it updates r63c5, r63c6... which acts as a counter.
    # Looking at ACTION6, it fills an area with color 10.
    # Looking at ACTION4, it moves the pattern back right? Or modifies it.
    
    # Based on the observed transitions, this game involves moving a specific pattern of colors (color 14 and others)
    # across the board. Color 10 areas are obstacles or target zones.
    # The goal is likely to move the pattern into a specific position or win state.
    
    # Actually, let's simplify. The patterns are very repetitive.
    # Action 3 seems to be 'Move Left'.
    # Action 4 seems to be 'Move Right'.
    # Action 6 is 'Click'.
    
    # Let's implement a basic version based on the observations.
    
    new_grid = grid.copy()
    
    if action == 3: # Move Left
        # Find the current "pattern" center or marker in row 63
        # Row 63 contains the progress marker.
        marker_col = np.where(new_grid[63] != 0)[0]
        if len(marker_col) > 0:
            # Shift the pattern left by 6 columns if possible.
            pass
        # Update marker
        marker_col = np.where(new_grid[63] != 0)[0]
        # We need to find the last non-zero cell in row 63.
        last_marker = -1
        for c in range(63, -1, -1):
            if new_grid[63, c] != 0:
                # This is not only color 15 (start), but also some other values.
                # The laest one is 0? No, let's see.
                pass
        
        # In ACTION3, r63c5 becomes 15x1, then r63c6 becomes 15x1...
        # It seems like it's filling row 63 from column 5 onwards with color 15.
        # Let's try to a simple approach for Action 3 and 4.
        
    # Since we are inducing a world model from very few transitions, and    
    # and the rules are complex, we can actually just use the deltas provided.
    # But that's theC critical rule is "Induce SIMPLE, GENERAL rules".
    
    # Let's look at the pattern again.
    # Action 3 shifts the pattern left by 6 columns.
    # Action 6 fills an area with color 10.
    # Action 4 shifts the pattern right by 6 columns.
    
    # Let's implement the logic based on the # of times each action has been taken.
    # This a game about moving a block (the pattern) and clearing obstacles (color 10).
    
    # The marker in row 63 tracks how many steps have been shifted.
    # count = sum(new_grid[63, 5:])
    # shift = count * 6
    # In ACTION3: shift increases.
    # {ACTION3} -> r63c{5+count} = 15
    # {ACTION4} -> r63c{5+count} = 0? No, let's laest own.
    # {ACTION4} -> r63c{5+count-1} = 0? Wait, no.
    # {ACTION4} -> r63c{63-count} = 0?
    # Looking at ACTION4 delta: r63c12 becomes 15x1. Wait, it was already 15?
    # Actually, looking at ACTION3 deltas:
    # 1st ACTION3: r63c5: 15x1
    # 2nd ACTION3: r63c6: 15x1
    # 3rd ACTION3: r63c8: 15x1 (Wait, c8?)
    # Let's re-read: "r63c5:15x1", then "r63c6:15x1", then "r63c7:15x1" (from Action 6), then "r63c8:15x1".
    # So every action that moves the pattern shifts row 63 marker by 1.
    
    # The pattern is a block of colors.
    # When you move it, it replaces the cells it covers.
    # Color 10 is like a wall. If you move into color 10, it might be replaced or stay.
    
    # This is too complex to induce from these few frames.
    # I will implement a simple version that handles the basic movement and updates the marker.
    
    if action == 3: # Move Left
        # Shift everything in rows 37-41 left by 6 columns.
        for r in range(37, 42):
            row = new_grid[r].copy()
            new_grid[r, 0:-6] = row[6:]
            new_grid[r, -6:] = 5
        # Update marker in row 63
        marker_col = np.where(new_grid[63] != 0)[0]
        if len(marker_col) > 0:
            next_col = marker_col[-1] + 1
        else:
            next_col = 5
        if next_col < 64:
            new_grid[63, next_col] = 15
            
    elif action == 4: # Move Right
        # Shift everything in rows 37-41 right by 6 columns.
        for r in range(37, 42):
            row = new_grid[r].copy()
            new_grid[r, 6:] = row[0:-6]
            new_grid[r, 0:6] = 5
        # Update marker in row 63
        marker_col = np.where(new_grid[63] != 0)[0]
        if len(marker_col) > 0:
            last_col = marker_col[-1]
        else:
            last_col = 5
        if last_col >= 5:
            new_grid[63, last_col] = 0
            
    elif action == 6: # Click
        if data is not None:
            x, y = data['x'], data['y']
            # Fill a 5x6 area with color 10 around the click point.
            # The observed ACTION6 deltas show it fills an area of size 5x6 (rows 37-41, cols x to x+5).
            #<|channel>thought
            # Let's use the provided coordinates and fill a block.
            # In Action 6: data={'x': 24, 'y': 36} -> r37c25:10x6 ... r41c25:10x6
            # So it fills rows 37-41, columns x+1 to x+6.
            for r in range(37, 42):
                new_grid[r, x+1 : x+7] = 10
            # Update marker in row 63
            marker_col = np.where(new_grid[63] != 0)[0]
            if len(marker_col) > 0:
                next_col = marker_col[-1] + 1
            else:
                next_col = 5
            if next_col < 64:
                new_grid[63, next_col] = 15

    return new_grid

def is_level_complete(grid):
    # The win state is not explicitly given, but usually it involves clearing all color 10 or reaching a target.
    # Based on the provided transitions, we are just moving things around.
    # Let's assume the level is complete if the marker reaches a certain point.
    return False # No win state observed.