import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action 4 is a right-shift of certain elements in the same row/column structure.
    # Based on observed transitions, ACTION4 shifts a "window" or "block" of cells to the right.
    # The shift distance is 3 units per call.
    # It affects rows 14-31 and columns that are shifted by 3 each time.
    # Specifically, it looks like it's shifting a pattern of colors [0, 10, 8, 14] across the board.
    
    new_grid = grid.copy()
    if action == 4:
        # Identify the region being shifted.
        # We observe that the shift happens in rows 14 to 31.
        # 
        # Let's analyze the specific delta changes.
        # In first transition: r14c11:10x3, r14c26:0x3...
        # This means at row 14, col 11 becomes 10, and col 26 becomes 0.
        #<|channel>thought
        # The logic seems to be a simple horizontal translation of a subset of the grid.
        # However, the most robust way to implement this for ARC is to detect the moving parts.
        # Looking at the deltas, ACTION4 shifts everything in the "active" area (rows 14-31) 
        # to the right by 3 pixels.
        
        # Define the active area based on observed transitions.
        # Rows 14 to 31 are heavily modified.
        # Row 0 also has single pixel changes (r0c16:0x1, etc.), which might be a cursor or marker.
        # Row 0 column index increases by 1 each time.
        
        # Shift region: rows 14 to 31.
        # We shift the content of these rows within the board width.
        # For each row in [14, 31], we move values from col j to col j+3.
        # The cells that were shifted into should now have the value they had before? 
        # No, it's more like a sliding window.
        
        # Let's try a simpler approach: find all non-background (non-10) cells in those rows and shift them.
        # But background is 10.
        
        # Based on the delta patterns:
        # r14c11:10x3, r14c26:0x3 -> Col 11 becomes 10, Col 26 becomes 0.
        # This means whatever was at col 11 moved to col 14, etc.
        # It looks like ACTION4 shifts the "empty" spaces (color 0) and other colors to the right by 3.
        
        # To implement this accurately without knowing the exact mask:
        # We can observe that for action 4, the grid effectively shifts its contents horizontally.
        # Since we only have one example of Action 4, let's use a general shift logic.
        
        # Shift row 0 marker:
        marker_col = np.where(grid[0] == 0)[0]
        if len(marker_col) > 0:
            new_grid[0, marker_col[0]] = 4 # restore old marker
            new_grid[0, marker_col[0] + 1] = 0 # move marker right
        else:
            # If no marker is found, try to find where it might be based on initial state
            # Initial state r0 has color 4 from c16 to c47. Marker starts at c16?
            pass

        # Shift rows 14-31:
        for r in range(14, 32):
            row = grid[r].copy()
            shifted_row = np.full(row.shape, 10) # Fill with background
            # The observed deltas show values moving right by 3.
            # We can simulate this by shifting the array and padding.
            # However, the boundaries are tricky. Let's just shift the whole row.
            shifted_row[3:] = row[:-3]
            new_grid[r] = shifted_row
            
        return new_grid

    return new_grid

def is_level_complete(grid):
    # Win state not provided, but usually involves reaching a target or clearing a board.
    # Given the pattern, maybe when the same colors align?
    # Without a win state example, we return False unless a specific condition is met.
    return False