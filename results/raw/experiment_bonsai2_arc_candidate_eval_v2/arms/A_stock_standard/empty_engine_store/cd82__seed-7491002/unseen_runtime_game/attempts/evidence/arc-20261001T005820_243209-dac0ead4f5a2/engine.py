import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    
    # Track a counter for the bottom-right corner decrement
    # The cell at (63, 63) starts at 5 and decrements by 1 on each action
    # We infer the current value from the grid itself
    
    if action == 6:
        # Click action - just decrement the counter
        px = data.get('x', 0)
        py = data.get('y', 0)
        # Check if clicking on specific interactive elements
        # For now, just handle the counter decrement
        pass
    elif action == 5:
        # Up movement / gravity up
        # Objects fall upward? Or player moves up?
        # Looking at transitions: ACTION5 causes 15s to move in certain areas
        # r34c27:15x10 etc - the 0-block area gets filled with 15
        # This seems like a "fill" or "move up" mechanic
        pass
    elif action == 2:
        # Right movement
        pass
    elif action == 4:
        # Left movement  
        pass
    elif action == 3:
        # Down movement
        pass

    # The consistent pattern across ALL actions is that g[63][63] decreases by 1
    # Let's check: initial has r63:4x64 (all 4s). 
    # After first action: r63c63:5x1 -> cell(63,63) becomes 5
    # Wait, that's an INCREASE from 4 to 5.
    
    # Let me re-read: INITIAL r63:4x64 means all cells are 4.
    # Transition 1 delta: r63c63:5x1 means cell(63,63) changed TO 5.
    # Transition 2 delta: r63c62:5x1 means cell(63,62) changed TO 5.
    # Transition 3 delta: r63c61:5x1 means cell(63,61) changed TO 5.
    # ...
    # So it's filling row 63 from right to left with 5s!
    
    # Each action fills one more cell in row 63 from the right side with value 5
    # Find the current rightmost unfilled position
    
    # Count how many 5s are already at the end of row 63
    row63 = g[63]
    fill_count = 0
    for c in range(63, -1, -1):
        if row63[c] == 5:
            fill_count += 1
        else:
            break
    
    next_col = 63 - fill_count
    if next_col >= 0:
        g[63][next_col] = 5

    return g


def is_level_complete(grid):
    # Check if row 63 is fully filled with 5s (all 64 cells)
    # Or some other win condition
    # Based on observed data, no win state was reached in the transitions shown
    # The level seems to complete when all interactive elements are resolved
    # For now, check if row 63 is completely 5s
    if np.all(grid[63] == 5):
        return True
    return False