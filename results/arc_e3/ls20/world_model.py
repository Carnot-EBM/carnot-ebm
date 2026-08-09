import numpy as np

def engine(grid, action, data):
    """
    Predicts the next state of the grid given an action.
    The game involves moving a 5x5 block consisting of two rows of color 12 
    and three rows of color 9 within a region of background color 3.
    Movement occurs in increments of 5 pixels.
    """
    # Block definition: 5x5 area with top 2 rows = 12, bottom 3 rows = 9
    block_h, block_w = 5, 5
    
    # Find current position of the movable block
    # Search for the top-left corner (r, c) where the 5x5 pattern exists
    curr_r, curr_c = -1, -1
    for r in range(grid.shape[0] - block_h + 1):
        for c in range(grid.shape[1] - block_w + 1):
            match = True
            for dr in range(block_h):
                for dc in range(block_w):
                    expected_color = 12 if dr < 2 else 9
                    if grid[r + dr, c + dc] != expected_color:
                        match = False
                        break
                if not match: break
            if match:
                curr_r, curr_c = r, c
                break
        if curr_r != -1: break

    if curr_r == -1:
        return grid.copy()

    new_grid = grid.copy()
    dr, dc = 0, 0
    if action == 1: # Up
        dr = -5
    elif action == 2: # Down
        dr = 5
    elif action == 3: # Left
        dc = -5
    elif action == 4: # Right
        dc = 5
    else:
        return new_grid

    # Calculate new position
    next_r, next_c = curr_r + dr, curr_c + dc

    # Boundary checks to prevent index errors
    if not (0 <= next_r <= grid.shape[0] - block_h and 0 <= next_c <= grid.shape[1] - block_w):
        return new_grid

    # To move the block, we first clear its current position with background color 3
    for r in range(curr_r, curr_r + block_h):
        for c in range(curr_c, curr_c + block_w):
            new_grid[r, c] = 3

    # Then place the block at the new position
    for dr_off in range(block_h):
        for dc_off in range(block_w):
            color = 12 if dr_off < 2 else 9
            new_grid[next_r + dr_off, next_c + dc_off] = color

    return new_grid

def is_level_complete(grid):
    """
    The level is complete when the movable 5x5 block reaches the target center 
    at row 10, column 34.
    """
    # Target coordinates for the top-left corner of the 5x5 block
    target_r, target_c = 10, 34
    block_h, block_w = 5, 5
    
    if target_r + block_h > grid.shape[0] or target_c + block_w > grid.shape[1]:
        return False
        
    # Check if the block is positioned exactly at (10, 34)
    for r in range(target_r, target_r + block_h):
        for c in range(target_c, target_c + block_w):
            expected_color = 12 if (r - target_r) < 2 else 9
            if grid[r, c] != expected_color:
                return False
    return True