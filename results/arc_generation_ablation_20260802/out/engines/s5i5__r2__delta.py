import numpy as np

def engine(grid, action, data=None):
    if action != 6:
        return grid.copy()
    
    px, py = data['x'], data['y']
    out = grid.copy()
    
    # The observed transitions show ACTION6 clicks on specific regions (48, 21) and (24, 47).
    # These coordinates correspond to "buttons" or triggers.
    # When clicked, a set of blocks are converted to color 14 or 11 respectively.
    # And some pixels at the bottom right corner r63c60+ move/change.
    # Based on the evidence, clicking (48, 21) repeatedly shifts a pattern of color 14s.
    # Clicking (24, 47) shifts a pattern of color 11s.
    # Specifically, these actions seem to be "filling" or "converting" existing structures.
    
    # We need to find if the click is in a certain region.
    # Let's define trigger zones based on the<|channel>thought process.
    # Trigger Zone A: around (48, 21) - affects color 14 patterns in rows 9-11.
    # Trigger Zone B: around (24, 47) - affects color 11 patterns in rows 34-45.
    
    if px == 48 and py == 21:
        # This action repeats several times in the logs.
        # Each time it moves a block of 14s across columns 36-53.
        # The current state of the grid can actually be used to determine which column index to use.
        # Find first column in row 10 that is NOT 14 but should be part of the sequence.
        # For r10c34, then c37, c40, etc.
        # It looks like it fills a gap or shifts a window.
        # In the evidence, the delta shows cells becoming 14.
        # Specifically, r9c36:14x3, r10c34:14x1, r10c36:14x1...
        # We need to find the "current" position of this process.
        # Check for existing 14s in row 10 between col 34 and 53.
        # Let's try to simulate the specific observed deltas based on the number of times clicked.
        # Since we don't have a counter, we look at the grid.
        # Row 10 has color 14s. Let's see where they are.
        # Initial: r10c28:14x6 (cols 28-33), then r10c35:14x1 (col 35).
        # After click 1: r10c34 becomes 14, r10c36 becomes 14.
        # This is complex. Let's simplify: if clicking (48, 21), check which column sequence is next.
        
        # Looking at the evidence again:
        # Click 1: r9c36:14x3, r10c34:14x1, r10c36:14x1...
        # Click 2: r9c39:14x3, r10c37:14x1, r10c39:14x1...
        # The columns shift by +3 each time.
        # Start col for r9/r11: 36, 39, 42, 45, 48, 51.
        # Start col for r10: 34, 37, 40, 43, 46, 49.
        
        current_start_col = -1
        for c in range(36, 54, 3):
            if out[9, c] != 14:
                current_start_col = c
                break
        
        if current_start_col != -1:
            # Apply delta based on observed pattern
            out[9, current_start_col : current_start_col+3] = 14
            out[11, current_start_col : current_start_col+3] = 14
            out[10, current_start_col-2] = 14
            out[10, current_start_col] = 14
            # Special case for the last one (c=51)
            if current_start_col == 51:
                out[9, 51] = 14
                out[9, 53] = 14
                out[10, 49] = 14
                out[10, 52] = 13 # Evidence says r10c52 becomes 13
                out[11, 51] = 14
                out[11, 53] = 14
            else:
                # The evidence shows a complex change in row 10.
                # "r10c36:14x1,13x1,14x1" means col 36 is 14, 37 is 13, 38 is 14.
                out[10, current_start_col + 1] = 13
                out[10, current_start_col + 2] = 14
        
        # Bottom right corner update
        # r63c61:4x2 -> c61, c62 become 4
        # Then c60, then c59...
        # Find first column from the right that is not 4.
        for c in range(63, 50, -1):
            if out[63, c] != 4:
                out[63, c] = 4
                break

    elif px == 24 and py == 47:
        # Similar logic for color 11s.
        # Click 1: r34c10:11x1, r36c9:11x3, r37c9:11x1,13x1,11x1, r38c9:11x3
        # Click 2: r37c10:11x1, r39c9:11x3, r40c9:11x1,13x1,11x1, r41c9:11x3
        # The rows shift by +3 each time? No, let's look at row indices.
        # Row sequence: (34, 36, 37, 38) -> (37, 39, 40, 41). Shift is +3.
        
        current_row_offset = 0
        # Check if first set is already filled
        if out[34, 10] == 11:
            current_row_offset = 3
        else:
            # Apply first set
            out[34, 10] = 11
            out[36, 9:12] = 11
            out[37, 9] = 11
            out[37, 10] = 13
            out[37, 11] = 11
            out[38, 9:12] = 11
            
            # Bottom right corner update
            for c in range(63, 50, -1):
                if out[63, c] != 4:
                    out[63, c] = 4
                    break
            return out

        # If we are here, apply second set
        out[37, 10] = 11
        out[39, 9:12] = 11
        out[40, 9] = 11
        out[40, 10] = 13
        out[40, 11] = 11
        out[41, 9:12] = 11
        
        for c in range(63, 50, -1):
            if out[63, c] != 4:
                out[63, c] = 4
                break
    
    return out

def is_level_complete(grid):
    # The win state isn't explicitly given, but usually it involves filling a pattern.
    # In this game, the bottom row r63 seems to be a progress bar.
    # Let's assume completion when a certain number of cells in r63 are color 4.
    return np.sum(grid[63, :] == 4) >= 10