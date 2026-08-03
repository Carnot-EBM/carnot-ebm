import numpy as np

import numpy as np

def engine(grid, action, data):
    # Action 2: Move Down (observed shift of patterns)
    if action == 2:
        new_grid = grid.copy()
        # The pattern shifts down by 3 rows
        # We need to identify the 'active' blocks of colors 5 and 4
        # In the initial state, these are around r15-r23 and c9-c57
        # Shift everything except background color 9 and boundary color 10/11
        # Based on the same delta, wes see a 3x3 block movement
        # Specifically, the cells at [row, col] move to [row+3, col]
        # For simplicity, in this game, ACTION 2 seems to be "Move Down"
        # Let's implement a general shift for non-background pixels
        mask = (grid != 9) & (grid != 10) & (grid != 11)
        # To actually match the deltas exactly, it is complex.
        # But based on the observed transitions, the 3x3 blocks of 5s and 4s move down.
        # shifted_mask = np.roll(mask, 3, axis=0)
        # Instead of let's just do a a simple row shift for specific columns
        # new_grid[mask] = 9
        # # This is<|channel>thought
        # return new_grid
        
        # Looking closer at the deltas:
        # r18c3:9x9 -> means rows 18,19,20 cols 3-11 become color 9
        # The pattern shifts by 3 units.
        # We identify all regions of colors 4, 5, 0
        # Shift them down by 3.
        # Background is 9.
        # Color 10 is a vertical wall.
        # Color 11 is right boundary.
        
        # Create a copy to avoid modifying original grid while iterating
        res = grid.copy()
        
        # Find all pixels that are not background (9), walls (10), or boundaries (11)
        # These are the "objects" (colors 0, 4, 5)
        obj_mask = (grid != 9) & (grid != 10) & (grid != 11)
        
        # To simulate the movement, we first clear the objects from their current positions
        # and then place them in their new positions.
        # However, the observed transitions show specific blocks moving.
        # Let's implement a shift of 3 rows for these objects.
        
        # For each object pixel, move it to row + 3
        # If it hits a wall (color 10), it might stop or wrap.
        # In the deltas, they just move down.
        
        # Clear old positions
        res[obj_mask] = 9
        
        # Place in new positions
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if obj_mask[r, c]:
                    new_r = r + 3
                    if new_r < grid.shape[0]:
                        # Only move if target is not a wall/boundary
                        if res[new_r, c] == 9:
                            res[new_r, c] = grid[r, c]
        return res

    # Action 3: Move Left/Right (observed horizontal shifts)
    if action == 3:
        res = grid.copy()
        obj_mask = (grid != 9) & (grid != 10) & (grid != 11)
        res[obj_mask] = 9
        
        # Observed ACTION 3 moves objects horizontally by some amount (e.g., 3 columns)
        # Let's assume it's "Move Right" or "Move Left" based on the delta
        # In first transition, r15c6 becomes 5x3 (was r15c9). Shift of -3?
        # No, let's look at r15c6:5x3 vs initial r15c9:5x9.
        # It looks like a shift of -3 columns.
        
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if obj_mask[r, c]:
                    new_c = c - 3
                    if new_c >= 0:
                        if res[r, new_c] == 9:
                            res[r, new_c] = grid[r, c]
        return res

    return grid

def is_level_complete(grid):
    # Win state not provided, but usually involves clearing objects or reaching a goal.
    # Based on common ARC patterns, we check if all object pixels are gone.
    obj_mask = (grid != 9) & (grid != 10) & (grid != 11)
    return not np.any(obj_mask)

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition for ar25 is that all cells are the same color (excluding background).
    """
    grid = np.array(grid)
    # Find all non-zero colors
    non_zero_colors = np.unique(grid[grid != 0])
    
    # If there are no non-zero colors, it's not a win state
    if len(non_zero_colors) == 0:
        return False
    
    # If all non-zero cells are the same color, it's a win state
    return len(non_zero_colors) == 1
