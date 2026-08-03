import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, ACTION3 seems to move a specific pattern of blocks
    # and potentially clear some other blocks (color 14 -> 0).
    # The pattern being moved is [9, 10, 2, 2] which appears in rows 19 and 20.
    # In each transition, the pattern moves left by 2 columns.
    # Move the pattern [9, 10, 2, 2] in rows 19 and 20.
    new_grid = grid.copy()
    if action == 3:
        # Find the leftmost occurrence of the pattern [9, 10, 2, 2] in row 19
        # We look for where it starts.
        for c in range(64):
            if (c + 3 < 64 and 
                grid[19, c] == 9 and 
                grid[19, c+1] == 10 and 
                grid[19, c+2] == 2 and 
                grid[19, c+3] == 2):
                
                # Save current position
                start_col = c
                # Clear old position
                new_grid[19, start_col:start_col+4] = 5 # Background color
                new_grid[20, start_col:start_col+4] = 5 # Background color
                
                # Move to new position (left by 2)
                new_col = max(0, start_col - 2)
                new_grid[19, new_col:new_col+4] = [9, 10, 2, 2]
                new_grid[20, new_col:new_col+4] = [9, 10, 2, 2]
                break
        
        # Also clear some cells of color 14 on the right edge.
        # The observed transitions show r6c62:0x2, r7c62:0x2 etc. which are color 14.
        # The pattern movement is not always paired with clearing.
        # We'll try to simulate a simple rule: for every ACTION3, 
        # we're moving the block and potentially clearing two rows of color 14s.
        # Let's check if there's any other logic.
        # In the first transition, only move occurred.
        # In second, r6,r7 cleared.
        # In third, r8,r9 cleared.
        # In fourth, only move occurred.
        # In fifth, r10,r11 cleared.
        # This seems like it happens every other time or based on some position.
        # For instance, when start_col was 37, then 35, then 33, then 31, then 29...
        # It looks like the moves are consistent (move left by 2).
        # The la-la-la-//-la-la-la sequence suggests a<|channel>thought process.
        # But since we don'tt have enough data to actually see the "win" state,
        # we'll implement the basic movement.
        
        # To handle the clearing of color 14s more accurately:
        # We look at which ones were cleared in order.
        # (6,7), (8,9), (10,11), (12,13), (14,15)
        # Let's track this as a internal state? No, engine must be pure.
        # We can derive the current row index for clearing from the block' same column.
        # If start_col is 37, move to 35. Then if start_col is 35, clear (6,7) and move to 33.
        # If start_col is 33, clear (8,9) and move to 31.
        # If start_col is 31, only move to 29.
        # This is slightly inconsistent. Let's just do the movement.
        
    return new_grid

def is_level_complete(grid):
    # Win state not provided, but typically it involves moving something or clearing everything.
    # For now, return False unless we see a specific condition.
    return False