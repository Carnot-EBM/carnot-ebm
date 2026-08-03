import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where objects move based on actions.
    # Based on the observed transitions, it's a bit complex but doesn't explicitly show a win state.
    # Let's implement a basic movement logic for the same object that might be<|channel>thought
    # and potentially some interaction with walls/obstacles.
    
    new_grid = grid.copy()
    
    if action == 4: # ACTION4 usually corresponds to 'right' in these games
        # Find all cells of color 9 (the moving object)
        # We see color 9 shifting right by 3 columns in each transition.
        # Shift everything that is not background (color 5)
        # To simplify, we actually shift specific colors or target areas.
        # In the observations, only certain rows are shifted.
        #
        # Looking at the delta: r11c39:5x1 r11c42:9x1 ...
        # This means cell (11, 39) becomes 5, and (11, 42) becomes 9.
        # A shift of +3.
        
        mask = (grid != 5)
        # Since it's a 64x64 grid, we need to handle boundaries.
        shifted_mask = np.roll(mask, 3, axis=1)
        
        # For the same logic, let's apply it to the values.
        # We will create a temporary grid to store the result.
        temp_grid = np.full_like(grid, 5)
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if grid[row, col] != 5:
                    new_col = col + 3
                    if new_col < grid.shape[1]:
                        temp_grid[row, new_col] = grid[row, col]
                    else:
                        # Wrap around or stop? The data shows no wrap around for most cells.
                        pass
        
        # Now we must merge this with the background.
        # In ACTION4, some things are replaced by color 5 and others by their shifted value.
        # # Let's refine based on the delta.
        # r24c26:5x3 r24c39:9x1 r24c42:0x1 ...
        # This is complex. Let's try a simpler approach: shift all non-5 colors right by 3.
        
        # To avoid overwriting, we iterate backwards.
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]-1, -1, -1):
                if grid[row, col] != 5:
                    val = grid[row, col]
                    if col + 3 < grid.shape[1]:
                        new_grid[row, col+3] = val
                    new_grid[row, col] = 5
        return new_grid

    if action == 1: # ACTION1 usually corresponds to 'up' or similar.
        # Looking at the delta for ACTION1:
        # r13c21:11x1 r14c21:11x1 r15c21:11x1...
        # It seems like it shifts some objects vertically.
        # The rows are shifting up (e.g., from 13, 14, 15 down to 10, 11, 12).
        # Let's try a vertical shift of -3.
        
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if grid[row, col] != 5:
                    val = grid[row, col]
                    if row - 3 >= 0:
                        new_grid[row-3, col] = val
                    new_grid[row, col] = 5
        return new_grid

    if action == 5: # ACTION5 is often an interaction/collect action.
        # In the observations, only one cell changes color.
        # This might be "completing" a part of the puzzle.
        pass

    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it involves clearing certain colors or reaching a goal.
    # Based on the INITIAL GRID and transitions, we can check if any specific condition is met.
    # Since no WIN STATE was given, let's return False unless something obvious happens.
    # The r63c59 area seems to be a counter (color 1).
    # Let's assume the level is complete when that counter reaches a certain value.
    # However, without more data, this is a guess.
    # A common pattern in these games is checking for the absence of some objects.
    # Check if all non-background cells are gone? Or some target reached.
    # For now, just return False as no win state was observed.
    count_non_bg = np.sum(grid != 5)
    return count_non_bg == 0