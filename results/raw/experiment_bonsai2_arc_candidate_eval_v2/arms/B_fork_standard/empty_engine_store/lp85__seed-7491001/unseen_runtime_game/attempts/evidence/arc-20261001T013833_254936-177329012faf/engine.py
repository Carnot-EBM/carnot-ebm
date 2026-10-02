import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    if action != 6 or data is None:
        return g
    
    px, py = data['x'], data['y']
    
    # Determine which side was clicked (left=col<32, right=col>=32)
    left_side = px < 32
    
    # Define the 4 rows of blocks and their column positions
    row_defs = [
        (19, 22),
        (25, 28),
        (31, 34),
        (37, 40),
    ]
    
    col_positions = [12, 18, 24, 30, 36, 42, 48]
    
    for r_start, r_end in row_defs:
        current_vals = []
        for c in col_positions:
            val = int(g[r_start, c])
            current_vals.append(val)
        
        new_vals = list(current_vals)
        if left_side:
            new_vals[0], new_vals[-1] = new_vals[-1], new_vals[0]
        else:
            new_vals[1], new_vals[-2] = new_vals[-2], new_vals[1]
        
        for i, c in enumerate(col_positions):
            for r in range(r_start, r_end + 1):
                g[r, c:c+4] = new_vals[i]
    
    # Update the indicator on the left border (col 0)
    # The indicator is a vertical bar of color 5 that moves down by 5 each click
    # Find current position of the indicator
    indicator_rows = np.where(g[:, 0] == 5)[0]
    if len(indicator_rows) > 0:
        top_row = int(indicator_rows.min())
        bottom_row = int(indicator_rows.max())
        height = bottom_row - top_row + 1
        
        # Clear old indicator
        g[top_row:bottom_row+1, 0] = 14
        
        # Move down by 5
        new_top = top_row + 5
        new_bottom = new_top + height - 1
        
        # Wrap around if it goes off screen
        if new_bottom >= 64:
            overflow = new_bottom - 63
            new_top -= (height + overflow)
            new_bottom = new_top + height - 1
        
        if new_top < 0:
            new_top = 0
            new_bottom = height - 1
        
        g[new_top:new_bottom+1, 0] = 5
    
    return g

def is_level_complete(grid):
    # Check if all blocks are in their "solved" positions
    # Based on observed patterns, the win state likely requires specific arrangements
    # For now, check if there's a consistent pattern that indicates completion
    # A simple heuristic: check if certain key cells have expected values
    # Since we don't have an explicit win state grid, use a conservative check
    return False