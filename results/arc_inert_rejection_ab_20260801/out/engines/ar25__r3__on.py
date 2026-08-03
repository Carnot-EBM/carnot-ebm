import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a movement-based puzzle where objects move in response to keyboard actions.
    # Based on the observed transitions, ACTION 2 and 3 are movements.
    # ACTION 2 moves things 'down' or shifts them.
    # ACTION 3 moves things 'right' or 'left'.
    # ACTION 6 is usually click.
    
    # This specific level layout suggests a symmetrical pattern of blocks that move.
    # We actually need to implement the logic for the same movement patterns seen in the observations.
    # Since we can<|channel>t induce a general rule easily from very few frames, 
    # let's look at the same cells changing.
    # Let's map the delta changes exactly if they possible.
    # But the rules must be general.
    # In this case, it's likely a "Sokoban" style movement where colors shift.
    
    new_grid = grid.copy()
    
    if action == 2: # Down
        # Shift everything down by 1? No, not quite.
        # It looks like certain clusters of color 5 and 4 are moving together.
        # And some markers (color 11) are shifting too.
        # The deltas show rX cY becoming something else.
        # For example, Action 2 shifted rows 15-20 to 18-24 etc.
        # Basically, a vertical shift of objects.
        shift = 1
        # Identify all non-background (9) cells.
        # Find coordinates of all pixels that aren't background.
        mask = (grid != 9)
        coords = np.argwhere(mask)
        
        # We move these coords down.
        for r, c in coords:
            # Check boundary
            if r + shift < grid.shape[0]:
                # To avoid overwriting, we do it on a copy
                pass

    # Since the actual logic is complex and I must provide an executable model,
    # and based on the provided transitions, ACTION 3 moves things horizontally 
    # and ACTION 2 moves things vertically.
    # Let's implement a simple "block movement" where blocks of colors 4, 5, 10, 11 move.
    
    # This looks like a game where you move a 'cursor' or 'player' and push blocks.
    # But there are multiple moving parts.
    # Looking at the deltas again:
    # Action 3: r15c6->r15c3 (left shift), but also some other changes.
    # Action 2: r15-17 -> r18-20 (down shift).
    
    # It seems that action 2 shifts objects DOWN and action 3 shifts objects LEFT/RIGHT.
    # However, without more data, this is hard to generalize.
    # Given the constraints, let's assume standard directional movements for these actions.
    # ACTION 2 = Down, ACTION 3 = Left? No, looking at the first ACTION 3 delta, it shifted c6 to c3, so LEFT.
    # Wait, the second ACTION 3 shifted c3 to something else... no, it says r15c3:5x3.
    
    # Actually, look at the very same cells changing in each sequence.
    # The "changed cells" are exactly what we need to apply.
    # Let's try to implement the movement of all non-9 cells.
    
    if action == 2: # Down
        # Shift everything not background down by 1 row.
        temp_grid = np.full(grid.shape, 9)
        for r in range(grid.shape[0] - 1):
            for c in range(grid.shape[1]):
                if grid[r, c] != 9:
                    temp_grid[r+1, c] = grid[r, c]
        return temp_grid
    elif action == 3: # Left/Right (The data shows shifts in columns)
        # Shift everything not background left by 3 cols based on the delta (c6 -> c3).
        temp_grid = np.full(grid.shape, 9)
        for r in range(grid.shape[0]):
            for c in range(3, grid.shape[1]):
                if grid[r, c] != 9:
                    temp_grid[r, c-3] = grid[r, c]
        return temp_grid

    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a goal or clearing blocks.
    # In this game, we don't have a win state example, but typically it's when certain colors are gone or in place.
    # Let's assume it's complete if color 5 and 4 are aligned.
    return False