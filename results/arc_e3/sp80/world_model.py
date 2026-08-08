import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action 4 seems to be a movement action that shifts an object of color 9.
    # Based on the observed transitions, ACTION4 moves the object of color 9 (shape 809ea44c8998d207)
    # by some amount in the x-direction. Each call to ACTION4 shifts it by 4 pixels.
    # Find the object of color 9.
    next_grid = grid.copy()
    if action == 4:
        # Identify the bounding box of the object of color 9
        rows, cols = np.where(grid == 9)
        if len(rows) > 0:
            # Assuming there is only one such object and typically rectangular
            y0, y1 = rows.min(), rows.max()
            x0, x1 = cols.min(), cols.max()
            # Shift by 4 units to the right
            # Fill old position with background color (color 12)
            next_grid[y0:y1+1, x0:x1+1] = 12
            # New position shifted by 4
            nx0, nx1 = x0 + 4, x1 + 4
            # Clip new position to board boundaries
            if nx0 >= 0 and nx1 < 64:
                next_grid[y0:y1+1, nx0:nx1+1] = 9
            else:
                # a simple clipping or just<|channel>thought
                pass
    elif action == 5:
        # Action 5 triggers a level completion/transition.
        # In the observed data, it changes the grid significantly.
        # laids out the next level's layout.
        # The provided "WIN TRANSITION" says applying ACTION5 completes the level.
        # We don't need to actually simulate the transition to the next level' same-way
        # # because is_level_complete will be checked on the result of engine(GRID_BELOW, 5).
        # Since we are only simulating one level, we can't find the exact rules for the 
        # transition to the next level. However, we look at what makes the move winning.
        # Let's assume Action 5 is some kind of 'submit' or 'activate' button.
        return next_grid # Return as current grid if we not know how to change it.
    
    return next_grid

def is_level_complete(grid):
    # Based on the observed transitions, the win state occurs when an object of color 9
    # is moved into a specific target area.
    # Looking at the GRID BEFORE THE COMPLETING ACTION, the object of color 9 (bbox=(16, 24, 19, 43))
    # has x0=24 and x1=43. In the INITIAL GRID, it was at x0=12, x1=31.
    # It shifted by 12 units total (three ACTION4 calls).
    # The same action sequence: ACTION4 -> ACTION4 -> ACTION4 then ACTION5.
    # The result of engine(GRID_BELOW, 5) must return True for is_level_complete.
    # The "WIN TRANSITION" says applying ACTION5 completes the level.
    # laids out the next level's layout.
    # The object of color 9 was at x0=24 in the final grid before Action 5.
    # The<|channel>thought process suggests that there are some targets or goals.
    # Let's assume the goal is to move the object of color 9 to a minimum x-coordinate of 24.
    return np.any(grid == 9) and np.where(np.where(grid == 9)[1].min() >= 24) # This logic is slightly wrong.
    # Correcting: check if any cell of color 9 is at x >= 24.
    # let's try a simpler condition: does the object of color 9 have its left edge at x=24?
    rows, cols = np.where(grid == 9)
    if len(cols) > 0:
        return cols.min() >= 24
    return False

# Re-evaluating is_level_complete based on the provided data:
# INITIAL GRID: obj4 (color 9) bbox=(16, 12, 19, 31). Left edge x=12.
# ACTION4 shifts it by 4 each time.
# After 3 ACTION4 calls: 12 + 4*3 = 24.
# Then ACTION5 completes the level.
# So the win condition is likely that the object of color 9 reaches x=24.

def is_level_complete(grid):
    rows, cols = np.where(grid == 9)
    if len(cols) > 0:
        return cols.min() >= 24
    return False