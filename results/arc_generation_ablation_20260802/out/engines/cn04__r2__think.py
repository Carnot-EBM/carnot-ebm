import numpy as np

def engine(grid, action, data):
    """
    The game 'cn04' appears to be a puzzle where ACTION4 (Right) moves 
    a set of blocks or a "cursor" structure across the grid.
    Looking at the deltas for ACTION4:
    - It shifts a pattern of color 10 and color 0 horizontally by 3 units each time.
    - The changes occur in specific row ranges (r14 to r31).
    - There is also a single cell change at r0 that increments its column index (c16, c17, c18, c19...).
    - This suggests a stateful movement of a complex object.
    """
    next_grid = grid.copy()
    
    if action == 4: # Move Right
        # Based on observed transitions, we shift the modified regions right by 3 columns.
        # However, since this is an executable world model, we must implement the logic 
        # that reproduces the delta.
        
        # The most consistent part of the transition is the shifting of values.
        # We can simulate this by identifying the "active" block area and shifting it.
        # But given the complexity of the run-length deltas, the simplest rule is:
        # For rows 14-31, certain patterns are shifted.
        
        # Let's look at the r0 change: r0c16:0x1 -> r0c17:0x1 ...
        # This looks like a marker moving.
        marker_col = -1
        for c in range(64):
            if next_grid[0, c] == 0:
                marker_col = c
                break
        
        if marker_col != -1 and marker_col < 63:
            next_grid[0, marker_col + 1] = 0
            # Note: In the observations, only one cell changes to 0 at r0 per ACTION4.
            # It doesn't say the previous one returns to 10, but usually these markers move.
            # However, looking closely at 'r0c16:0x1', then 'r0c17:0x1', etc., 
            # they are just setting that specific cell to 0.
            
        # The bulk of the movement happens between row 14 and 31.
        # We shift the content of columns [C : C+W] to [C+3 : C+3+W].
        # Since we don't have the full logic for which cells "clear" (become 10) 
        # and which "fill", we apply a general shift to the internal structure.
        
        # To be precise with the provided deltas:
        # Row 14-16: cols x..x+2 become 10, cols x+6..x+8 become 0
        # Row 17-19: cols x..x+2 become 10, cols x+3..x+5 stay/change, cols x+9..x+11 become 10...
        # This is essentially shifting a pattern right by 3.
        
        # Implementation of a simple horizontal shift for the active area:
        active_rows = range(14, 32)
        for r in active_rows:
            row_data = next_grid[r, :].copy()
            # Shift everything that isn't the background color (10) or boundary
            # In this specific game, it looks like columns 11 through 47 are moving.
            next_grid[r, 14:] = row_data[11:-3] # Simplified shift
            
    return next_grid

def is_level_complete(grid):
    """
    The win state is not explicitly provided, but usually involves reaching a target 
    or clearing a pattern. Without a WIN STATE grid, we return False unless 
    a known condition is met.
    """
    return False