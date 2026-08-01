import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, ACTION3 seems to move a specific pattern of colors 
    # (9, 10, 2, 2) in rows 19 and 20 leftwards.
    # It also changes some values at column 62 and 63 (the far right edge).
    # The movement is likely related to a moving object/entity.
    # Let's induce a general rule for the same pattern.
    
    next_grid = grid.copy()
    if action == 3: # ACTION3 moves something left
        # Identify the "object" pattern in rows 19-20
        # Look for the sequence [9, 10, 2, 2] in those rows
        for r in [19, 20]:
            # Find where the sequence starts
            # Find all indices where grid[r, :] == 9
            indices = np.where(grid[r, :] == 9)[0]
            for idx in indices:
                # Check if it's part of the sequence [9, 10, 2, 2]
                if idx + 3 < grid.shape[1]:
                    if np.array_equal(grid[r, idx:idx+4], [9, 10, 2, 2]):
                        # Move it left by 2 columns
                        # Fill old position with background color (5)
                        next_grid[r, idx:idx+4] = 5
                        # Place new position
                        new_idx = max(0, idx - 2)
                        next_grid[r, new_idx:new_idx+4] = [9, 10, 2, 2]
                        break # Only one such object per row
    
    # The right edge changes are likely a side effect or a timer/counter.
    # We can see r6c62:0x2, r7c62:0x2 etc. which means cells at col 62 and 63 become 0.
    # ACTION3 is repeated many times. Each time it takes some rows on the right edge.
    # Let's simulate this "right edge" depletion up to a certain limit.
    # Since we canre-t-//
    # This is not quite a detailed model but captures the movement of the pattern own.
    return next_grid

def is_level_complete(grid):
    # No win state provided in observed transitions.
    # Usually, win states are defined by when an object reaches a target.
    # return True if grid contains no color 14 (the wall/border on the right).
    # return False for now as no same general rule was    //
    return False