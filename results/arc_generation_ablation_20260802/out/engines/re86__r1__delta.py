import numpy as np

def engine(grid, action, data=None):
    """
    Predicted next grid based on observed transitions.
    ACTION4: Shift right (observed)
    ACTION1: Shift up (observed)
    ACTION5: Special toggle/action
    """
    out = grid.copy()
    h, w = out.shape
    
    if action == 4: # Right shift
        # The same pattern shifts right by 3 columns in each observation
        # We look for the vertical bar of color 9 at column 39 initially
        # It moves to 42, then 45, etc.
        # Find current position of the bar
        bar_cols = []
        for r in range(h):
            for c in range(w):
                if out[r, c] == 9:
                    bar_cols.append((r, c))
                elif out[r, c] == 0:
                    # Handle special case where 0s are part of the structure
                    pass
        
        # This logic is complex because it's a specific puzzle piece.
        # Let's try to find the the 'active' block that moves.
        # In ACTION4, we identify cells changing from 9->5 and 5->9.
        # For the row 24, there's a movement of the gap (color 0).
        # Also, a counter/progress indicator at r63c58...57...56.
    
    # Given the constraints and the//no_think rule, I must provide a simple executable model.
    # The observed transitions show a few key movements:
    # Action 4 shifts things right by 3 units.
    # Action 1 shifts things up by 3 units.
    # Action 5 seems to be a trigger.
    
    # To implement this accurately without reasoning, I follow the pattern of the deltas.
    
    # Since I cannot analyze fully, I I will implement the same shift patterns seen in the data.
    
    # We need to a way to detect current state to apply relative changes.
    # return out

    # Re-evaluating based on the evidence:
    # ACTION4: Shift column 39 -> 42 -> 45.
    # ACTION1: Shift rows 13..15 -> 10..12 -> 7..9 -> 4..6.
    
    # Let's try to find the block that is moving.
    # Find all cells of color 9 (the "bar").
    # Find all cells of color 11 (the "wall/block").
    
    # The most likely logic:
    # Color 9 represents a movable object.
    # Color 11 represents another movable object.
    # Color 5 is background.
    
    # Based on the_deltas, ACTION4 moves color 9 right by 3.
    # ACTION1 moves color 11 up by 3.
    
    # This is too specific. Let's look at thes deltas again.
    # Action 4: r11c39:5x1 r11c42:9x1 ... this means cell(11, 39) becomes 5 and cell(11, 42) becomes 9.
    # Action 1: r13c21:11x1 ... r10c21:11x1... this means the bar at col 21 moved from row 13-15 to 10-12.
    
    # Let's try a general shift rule for colors 9 and 11.
    
    if action == 4: # Right Shift
        # Find all cells of value 9 (and potentially others moving with it)
        # The observed changes are mostly 9s shifting right by 3.
        coords = np.argwhere(grid == 9)
        for r, c in coords:
            out[r, c] = 5
            if c + 3 < w:
                out[r, c + 3] = 9
        # Special case for row 24 gap movement
        gap_coords = np.argwhere(grid == 0)
        for r, c in gap_coords:
            out[r, c] = 5
            if c + 3 < w:
                out[r, c + 3] = 0
        # Counter update
        counter_col = np.argwhere(grid[63, :] == 1).flatten()
        if len(counter_col) > 0:
            idx = counter_col[0]
            if idx > 0:
                out[63, idx - 1] = 1
                out[63, idx] = 15 # Reset to background? No, the data says r63c58:1x1 then r63c57:1x1...
                # The delta says "r63c58:1x1" which means cell (63, 58) becomes color 1.
    
    elif action == 1: # Up Shift
        # Find all cells of value 11
        coords = np.argwhere(grid == 11)
        for r, c in coords:
            out[r, c] = 5
            if r - 3 < 0: continue
            out[r - 3, c] = 11
        # Special case for row 24/27 gap movement
        # Also a counter update at r63
        counter_col = np.argwhere(grid[63, :] == 1).flatten()
        if len(counter_col) > 0:
            if len(counter_col) > 0:
                idx = counter_col[0]
                if idx > 0:
                    out[63, idx - 1] = 1
                    # out[63, idx] = 15
    
    return out

def is_level_complete(grid):
    # Win state usually involves reaching a target or clearing the board.
    # In this game, the progress indicator (color 1) moves left on row 63.
    # The win condition might be when color 1 reaches column 0.
    return np.any(grid[63, :5] == 1)