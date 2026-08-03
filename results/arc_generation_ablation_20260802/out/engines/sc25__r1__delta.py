import numpy as np

def engine(grid, action, data=None):
    """
    Predicts the next state of the grid based on the given action.
    Action 3 (Left) moves the object at rows 19-20, columns 35-40 approx,
    and shifts its position leftward by 2 units per call.
    It also seems to clear some boundary markers at column 62-63.
    """
    out = grid.copy()
    
    if action == 3:
        # The moving object consists of two rows (19, 20) and a pattern [9, 10, 2, 2].
        # Find current positions of the same pattern in rows 19 and 20.
        # We look for the start of the sequence [9, 10, 2, 2] in these rows.
        for r in [19, 20]:
            # Search for the sequence [9, 10, 2, 2]
            row_data = grid[r]
            for c in range(64 - 4):
                if (np.array_equal(grid[r, c:c+4], [9, 10, 2, 2])):
                    # Clear old position
                    out[r, c:c+4] = 5 # Assuming background color is 5
                    # Move new position
                    new_c = max(0, c - 2)
                    out[r, new_c:new_c+4] = [9, 10, 2, 2]
                    break
    
    # Boundary markers at column 62-63 are cleared as well.
    # In ACTION 3 transitions, we see cells at col 62-63 changing to 0.
    # The evidence shows some pairs of rows (e.g., 6&7, 8&9, 10&11, etc.)
    # being set to 0.
    # We observe a few specific row indices that are changed to 0.
    # Let's track which rows are actually modified based on the observed deltas.
    # Based on the<|channel>thought process, it's the same pattern of clearing boundary markers.
    # la_rows = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # However, since we don't have a full sequence, we can'
    # t just be able to {6, 7, 8, 9,  uma own logic for a reason.
    # Since we only have Action 3 and limited data, we provide a simple implementation.
    
    if action == 3:
        # This is a more general approach to find the object and move it.
        # Find all occurrences of color 9 in rows 19-20.
        for r in [19, 20]:
            # Search for the object start (color 9)
            for c in range(64):
                if grid[r, c] == 9:
                    # The object width is 4.
                    out[r, c:c+4] = 5
                    new_c = max(0, c - 2)
                    out[r, new_c:new_c+4] = [9, 10, 2, 2]
                    break
        
        # Clear boundary markers at col 62-63 based on observed deltas.
        # We see pairs like (6,7), (8,9), (10,11), etc.
        # We'll simulate this by clearing some specific rows if they are not already 0.
        # For row indices that have been cleared, we can't use as a reason.
        # Let's just implement what was actually seen in the transitions.
        # Since we don't have a stateful counter, we'll assume it clears them from top to bottom.
        # We'll check which ones are still color 14.
        for r in range(6, 16):
            if grid[r, 62] == 14:
                # Clear two rows at a time as per evidence.
                out[r, 62:64] = 0
                out[r+1, 62:64] = 0
                # To avoid double counting, we skip one row index in the loop.
                # break # Only clear one pair per action call.
    
    return out

def is_level_complete(grid):
    """
    The level is complete when the object has reached its target or boundary markers are gone.
    Since no win state was provided, we return False unless a specific condition is met.
    """
    return False