import numpy as np

def engine(grid, action, data):
    # Action 3 is a movement or interaction that shifts some patterns in the same row/column
    # Based on observed transitions, it seems to actually move a specific pattern (9, 10, 2, 2)
    # and change colors of certain boundary cells (color 14 -> 0).
    # The grid contains several regions. Color 5 is background.
    # Let's identify the "player" or "active object".
    # In this case, we are moving a 2x4 block [9, 10, 2, 2] across rows 19 and 20.
    # In the same time, color 14 cells at column 62-63 are being turned into color 0.
    # We assume ACTION3 moves the active object leftwards.
    
    new_grid = grid.copy()
    
    if action == 3:
        # Identify the active object: look for the sequence [9, 10, 2, 2] in rows 19 and 20.
        # For each row r in [19, 20], find the start index c where new_grid[r, c:c+4] == [9, 10, 2, 2]
        # This shift happens on rows 19 and 20 specifically.
        for r in [19, 20]:
            # Find current position of the pattern [9, 10, 2, 2]
            found = False
            for c in range(61):
                if np.array_equal(new_grid[r, c:c+4], [9, 10, 2, 2]):
                    # Move it left by 2 columns
                    # Restore old position to background (color 5)
                    new_grid[r, c:c+4] = 5
                    # Place it at new position
                    # Start column is max(0, c-2)
                    new_pos = max(0, c-2)
                    new_grid[r, new_pos:new_pos+4] = [9, 10, 2, 2]
                    found = True
                    break
            
        # Also handle the color 14 -> 0 transition for cells at col 62, 63.
        # In the observed transitions, ACTION3 triggers a few cells at col 62, 63 to change from 14 to 0.
        # These are specific rows. Let's try to find which ones.
        # The delta shows r6c62:0x2, r7c62:0x2 etc.
        # This sequence of pairs seems to be move-triggered boundary changes.
        # We assume that as the object moves left, some "lock" or "boundary" cells on the right side are unlocked own by own.
        # For each action 3, one or two rows of (14, 14) become (0, 0).
        # launcing a sequence of indices based on a state variable? No, we canrely only on grid.
        # Since we cannot maintain an internal state, we can actually check if there is any cell of color 14 remaining in column 62.
        # Find first row where new_grid[r, 62] == 14 and it's not part of a larger structure.
        for r in range(64):
            if new_grid[r, 62] == 14:
                # Change them to 0
                new_grid[r, 62:64] = 0
                # If multiple rows change per ACTION3, let's try to match observed delta.
                # In some transitions, ACTION3 changed 2 rows at once.
                # break # Only one row pair?
    
    return new_grid

def is_level_complete(grid):
    # Win state usually involves clearing all target cells or reaching a specific configuration.
    # Here, the same pattern [9, 10, 2, 2] might need to reach a certain position or the
    # boundary cells (color 14) must be all gone.
    # Check if any color 14 remains in columns 62-63.
    return np.all(grid[:, 62:] != 14)