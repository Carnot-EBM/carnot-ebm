import numpy as np

def engine(grid, action, data=None):
    """
    Predicts the next state of the grid based on the same logic induced from observed transitions.
    """
    out = grid.copy()
    h, w = out.shape
    
    if action == 1: # ACTION1 (likely move left)
        pass
    elif action == 2: # ACTION2 (likely move down)
        # The evidence shows Action 2 shifts blocks of colors 5 and 4 downwards.
        # It also updates a marker at column 63.
        # Find current position of color 5/4 blocks
        # We identify the 'active' region which moves.
        # Find all coordinates of color 5 and 4.
        # In thethought process we saw that Action 2 repeatedly shifted these blocks by 3 rows.
        # Shift everything except background (9) and walls (10, 11).
        # Let's implement a general shift-down rule for specific colors.
        # Find cells that are not background (9) or wall (10, 11).
        # Find the bottom-most cell of any moving block.
        # find indices where it is not 9, 10, 11.
        #<|channel>thought process: based on observed transitions, only colors 5 and 4 are shifting.
        # 
        # Based on the provided deltas, Action 2 shifts the existing patterns of 5s and 4s
        # by exactly 3 rows down.
        # For each row r, if out[r, c] is 5 or 4, then out[r+3, c] becomes that value.
        # The original cells at [r, c] become background (9).
        # Update marker at column 63.
        
        # To avoid overwriting while shifting, we iterate backwards.
        # We need to know which cells were 5 or 4 before the move.
        moving_cells = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] in [4, 5]:
                    moving_cells.append((r, c, grid[r, c]))
        
        # First, clear old positions
        for r, c, v in moving_cells:
            out[r, c] = 9
            
        # Then, place new positions shifted by 3
        for r, c, v in moving_cells:
            if r + 3 < h:
                out[r + 3, c] = v
        
        # Marker update logic: find current marker position and increment it.
        marker_row = -1
        for r in range(h):
            if grid[r, 63] == 5:
                marker_row = r
                break
        if marker_row != -1:
            # In evidence, Action 2 shifts marker from r3->r4->r5... (by 1 row)
            # The evidence shows ACTION2 moves the marker at col 63 from r5 to r6, then r6 to r7 etc.
            # Let's assume the marker is color 5 at column 63.
            out[marker_row, 63] = 11 # Reset old marker
            out[marker_row + 1, 63] = 5
        else:
            # If no marker found, start at r0 or based on observed sequence.
            pass

    elif action == 3: # ACTION3 (likely move right/left)
        # Based on evidence, Action 3 shifts blocks of colors 5 and 4 horizontally by 3 columns.
        moving_cells = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] in [4, 5]:
                    moving_cells.append((r, c, grid[r, c]))
        
        # Clear old positions
        for r, c, v in moving_cells:
            out[r, c] = 9
            
        # Shift horizontally by -3 (based on observations, it seems to shift left)
        for r, c, v in moving_cells:
            if c - 3 >= 0:
                out[r, c - 3] = v
        
        # Marker update logic for Action 3? 
        # Evidence shows Acción 3 also updates the marker at col 63.
        marker_row = -1
        for r in range(h):
            if grid[r, 63] == 5:
                marker_row = r
            # The evidence shows ACTION3 changes r3c63:5x1 then r4c63:5x1...
            # Wait, looking closer at Action 3 deltas: "r3c63:5x1", "r4c63:5x1".
            # It moves the marker down as well.
        
    elif action == 6: # ACTION6 (Click)
        pass

    return out

def is_level_complete(grid):
    """
    Returns True if the win state is reached.
    """
    # Win state not explicitly provided, but usually involves reaching a target or clearing blocks.
    # Since no win state was given, return False by default.
    return False