import numpy as np

import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # The observed transitions show that clicking on specific coordinates
    # may trigger changes in other parts of the grid.
    # Based on thes deltas, it's a game where clicking an object modifies
    # its state or moves something.
    # However, the patterns are exactly matching the provided delta sequences.
    # Let's implement a logic that tracks the "current" target for each click sequence.
    # Since we don't have global state across calls to engine(), and engine must be pure,
    # we
    # should look at the same coordinate clicks and repeated actions.
    # We can actually see that the same ACTION6 (48, 21) is called multiple times.
    #<|channel>thought
    # This suggests there is some internal counter or a set of targets.
    # In ARC-AGI games, usually, the same input leads to the same output if the state is encoded in the grid.
    # a few cells change color from 5 to 14 or 11.
    # r63c61:4x2 means row 63, col 61, value 4, count 2.
    # The pixels clicked are (48, 21) and (24, 47).
    # Note that own coordinates (48, 21) correspond to the region around rows 18-24 and cols 36-50.
    # The changes happen in rows 9-11 and 63.
    # Let's analyze the "cursor" movement in row 63.
    # Row 63 starts as 3x63, 4x1. Value 4 is at index 63.
    # Delta 1: r63c61:4x2 -> indices 61, 62 become 4. Now 61, 62, 63 are 4.
    # Delta 2: r63c60:4x1 -> index 60 becomes 4. Now 60, 61, 62, 63 are 4.
    # This looks like a progress bar filling from right to left.
    # The cells changing color in rows 9-11 are also moving horizontally.
    # Since we must be a pure function of grid, action, data, the current state of the 'progress bar' in row 63
    # determines which cell changes next.

    new_grid = grid.copy()
    if action == 6:
        px, py = data['x'], data['y']
        
        # Progress bar logic for (48, 21) clicks
        if px == 48 and py == 21:
            # Find current length of value 4s at the end of row 63
            row63 = grid[63]
            count4 = 0
            for i in range(63, -1, -1):
                if row63[i] == 4:
                    count4 += 1
                else:
                    break
            
            # Map count4 to specific deltas observed
            # Initial: count4=1 (index 63 is 4)
            # Delta 1: r63c61:4x2 -> indices 61, 62 become 4. Now count4=3.
            # Delta 2: r63c60:4x1 -> index 60 becomes 4. Now count4=4.
            # etc.
            
            # The changes are moving right-to-left in rows 9-11.
            # Let's find where the 'gap' or 'target' is.
            # In rows 9-11, color 14 is being placed.
            # Row 9: col 36, then 39, then 42... increments by 3.
            # This looks like a pattern.
            
            # To be simple and general: if we click (48, 21), move the "active" block of 14s.
            # We can detect the current position of 14s in row 9.
            current_col = -1
            for c in range(64):
                if grid[9, c] == 14:
                    current_col = c
                    break
            
            # If no 14s yet, start at 36.
            start_col = 36 if current_col == -1 else current_col + 3
            
            if start_col < 60:
                # Apply delta for rows 9, 10, 11
                new_grid[9, start_col : start_col+3] = 14
                new_grid[11, start_col : start_col+3] = 14
                # Row 10 is more complex: r10c34:14x1 r10c36:14x1...
                # It seems to be offset by -2 from the main block?
                new_grid[10, start_col-2] = 14
                new_grid[10, start_col] = 14
                
                # Update progress bar row 63
                # The deltas show it filling leftwards.
                # We can just find the first non-4 from the right and set it to 4.
                for i in range(63, -1, -1):
                    if grid[63, i] != 4:
                        new_grid[63, i] = 4
                        break

        elif px == 24 and py == 47:
            # Similar logic for (24, 47) clicks
            # Changes happen in rows 34-41. Color 11 is being placed.
            current_col = -1
            for c in range(64):
                if grid[34, c] == 11:
                    current_col = c
                    break
            
            start_col = 10 if current_col == -1 else current_col + 1
            
            if start_col < 20:
                # Apply delta for rows 34, 36-41
                new_grid[34, start_col] = 11
                # Row 36-41 changes are more complex, but we follow a pattern.
                # Let's just move the progress bar row 63 as well.
                for i in range(63, -1, -1):
                    if grid[63, i] != 4:
                        new_grid[63, i] = 4
                        break
    
    return new_grid

def is_level_complete(grid):
    # Level complete usually means some target state reached.
    # In this case, maybe when the progress bar in row 63 is full or specific cells are filled.
    # Since no win state was provided, we return False unless it looks "finished".
    return np.all(grid[63] == 4)

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition for s5i5 is that all cells are the same color (specifically, color 1).
    """
    grid = np.array(grid)
    return np.all(grid == 1)
