import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    new_grid = grid.copy()
    H, W = new_grid.shape
    
    if action == 1: # Up
        pass
    elif action == 2: # Down
        # Move a pattern of blocks down
        # The observed transitions for ACTION2 show a shift in y-coordinates by 3 rows.
        # It seems to be moving two vertical columns of patterns.
        # Find all non-background cells that are not part of the static boundaries (column 30-32)
        # and move them down.
        # We identify "moving" parts based on their colors.
        # In thiss specific level, it's color 5 and 4.
        # 
        # Let's implement a a simple translation of the same set of pixels.
        # Shift everything except column 30-32 and background (9) and boundary marker (11).
        # Mask for things that actually move.
        mask = (new_grid != 9) & (new_grid != 11) & (np.arange(W)[None, :] != 30) & \
                (np.arange(W)[None, :] != 31) & (np.arange(W)[None, :] != 32)
        
        # To avoid overwriting, we iterate from bottom up.
        for r in range(H - 1, -1, -1):
            for c in range(W):
                if mask[r, c]:
                    val = new_grid[r, c]
                    # Target position
                    nr, nc = r + 3, c
                    if nr < H:
                        # The observed delta shows cells changing to 9 (background) at old pos
                        # and taking values at new pos.
                        pass
    
    # Based on thes specific transitions, ACTION2 shifts patterns down by 3 rows.
    # # Let's refine this a shift of 3 is too simple. It' same as 3-row blocks.
    #<|channel>thought
    # la// This is a a bit more complex. We are moving "objects" defined by 3xN or Nx3 blocks.
    # 
    # Looking at the deltas for ACTION2:
    # r15c3:9x9 -> r18c3:9x9
    # r18c3:5x6 -> r21c3:5x6
    # r24c9:5x3 -> r27c9:5x3
    # etc.
    # All these changes are shifted by exactly 3 rows.
    
    # To implement this generally:
    # Find all pixels that aren't background(9), static boundary(10), or marker(11).
    # Shift them down by 3.
    # Move everything except col 30-32 (color 10) and column 63 (color 11).
    
    moving_mask = (new_grid != 9) & (new_grid != 10) & (new_grid != 11)
    
    if action == 2: # Down
        shift = 3
        temp_grid = new_grid.copy()
        # Clear moving parts from old positions
        for r in range(H):
            for c in range(W):
                if moving_mask[r, c]:
                    temp_grid[r, c] = 9
        # Place moving parts at new positions
        for r in range(H):
            for c in range(W):
                if moving_mask[r, c]:
                    nr, nc = r + shift, c
                    if nr < H:
                        temp_grid[nr, nc] = new_grid[r, c]
        return temp_grid

    elif action == 3: # Right/Left?
        # ACTION3 shifts patterns horizontally.
        # In the first transition, it shifted by some amount.
        # In the second, it shifted again.
        # The deltas show a horizontal shift of -3 or +3.
        # Let's try a shift of 3 columns.
        shift = 3
        temp_grid = new_grid.copy()
        moving_mask = (new_grid != 9) & (new_grid != 10) & (new_grid != 11)
        
        # Clear old positions
        for r in range(H):
            for c in range(W):
                if moving_mask[r, c]:
                    temp_grid[r, c] = 9
        # Place at new positions
        for r in range(H):
            for c in range(W):
                if moving_mask[r, c]:
                    nc = c + shift
                    if 0 <= nc < W:
                        # Avoid overwriting static boundaries col 30-32
                        if not (30 <= nc <= 32):
                            temp_grid[r, nc] = new_grid[r, c]
        return temp_grid

    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a target or clearing the board.
    # In this game, it's not provided, but typically it's when certain colors are aligned.
    # return False as no win state was given.
    return False