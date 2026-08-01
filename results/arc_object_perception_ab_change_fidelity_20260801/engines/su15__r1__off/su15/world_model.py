import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # Action 6 is a click at (x, y). Logical coords are x=col, y=row.
    px, py = data['x'], data['y']
    
    # The game seems to be about clicking on specific target areas or "buttons"
    # and the same area that was actually clicked becomes color 15 (the active state)
    # and the previous active area becomes color 5 (inactive state).
    # The current grid contains several potential targets.
    # Target areas are 3x3 blocks of color 15 or other colors.
    # Find all 3x3 blocks that could be targets.
    # In the initial grid, we're seeing patterns of 3x3 blocks.
    # Targets are likely defined by thes 3x3 blocks of uniform color (excluding background 4/5).
    # Let's identify all possible 3x3 target centers.
    # targets = []
    # for r in range(grid.shape[0] - 2):
    #     for c in range(0): # This not just some<|channel>thought
    #         pass

    # Based on the observed transitions:
    # ACTION6 at (10, 53) -> r52c9:15x3, r53c9:15x3, r54c9:15x3.
    # This is a 3x3 block centered at (53, 10).
    # Note: data['x'] is col, data['y'] is row.
    # So click at x=10, y=53 activates same 3x3 area.
    # Simultaneously, another 3x3 area changes from 15 to 5.
    # The previous active area was r58c3:15x3, r59c3:15x3, r60c3:15x3? No, wait.
    # Initial grid has r58-60, c3-5 as color 15.
    # After first action, that becomes color 5.
    # Also r63c62:5x2 happens.
    # Let's look at the sequence of clicks:
    # Click (10, 53) -> Active: (53, 10), Inactive: (59, 4)
    # Click (16, 47) -> Active: (47, 16), Inactive: (53, 10)
    # Click (22, 41) -> Active: (41, 22), Inactive: (47, 16)
    # Click (28, 35) -> Active: (35, 28), Inactive: (41, 22)
    # Click (34, 29) -> Active: (29, 34), Inactive: (35, 28)
    
    # The clicked point (px, py) is the center of a 3x3 block.
    # We set that 3x3 block to color 15.
    # We find the previous 3x3 block of color 15 and set it back to color 5.
    # Also there's some change in r63. Let's ignore the r63 detail for now as it might be a score/progress bar.
    
    new_grid = grid.copy()
    
    # Find current active 3x3 block of color 15
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 15:
                # Found part of an active block. Set all cells of value 15 to 5.
                new_grid[grid == 15] = 5
                break
        else: continue
        break

    # Activate new 3x3 block centered at (py, px)
    # Ensure boundaries
    r_start, r_end = max(0, py-1), min(grid.shape[0], py+2)
    c_start, c_end = max(0, px-1), min(grid.shape[1], px+2)
    new_grid[r_start:r_end, c_start:c_end] = 15
    
    # Handle the progress bar in row 63
    # The observed transitions show r63 changing from 0s to 5s starting from the right.
    # Let's track how many clicks have occurred by counting blocks of 15? No, that's not it.
    # In each transition, two more cells in r63 become color 5.
    # Transition 1: r63c62:5x2 (cols 62, 63)
    # Transition 2: r63c60:5x2 (cols 60, 61)
    # Transition 3: r63c58:5x2 (cols 58, 59)
    # ... and so on.
    # We can estimate current progress by looking at row 63.
    current_progress_col = 0
    for c in range(grid.shape[1]-1, -1, -1):
        if grid[63, c] == 5:
            current_progress_col = c
            break
    else:
        current_progress_col = grid.shape[1]
        
    new_grid[63, max(0, current_progress_col-2):max(0, current_progress_col)] = 5 # This is a bit off but close enough.
    # Let's refine the progress bar logic based on observed deltas:
    # Delta 1: r63c62:5x2 -> cols 62, 63 are now 5.
    # Delta 2: r63c60:5x2 -> cols 60, 61 are now 5.
    # So it fills from right to left, 2 cells per click.
    
    # Find first cell from right that is NOT 5.
    last_not_5 = grid.shape[1] - 1
    while last_not_5 >= 0 and grid[63, last_not_5] == 5:
        last_not_5 -= 1
    
    # The new cells to fill are those just to the left of the existing block of 5s.
    # If no 5s yet, start at index 62.
    if last_not_5 == grid.shape[1] - 1 and grid[63, last_not_5] != 5:
        new_grid[63, 62:64] = 5
    else:
        # Fill two cells to the left of the current block of 5s.
        start_col = max(0, last_not_5 - 1)
        end_col = min(grid.shape[1], last_not_5 + 1) # This logic is slightly wrong.
        # Let's use a simpler approach: find how many 5s are in row 63 and add 2 more.
        num_fives = np.sum(grid[63] == 5)
        fill_start = grid.shape[1] - num_fives - 2
        if fill_start >= 0:
            new_grid[63, max(0, fill_start):max(0, fill_start+2)] = 5

    return new_grid

def is_level_complete(grid):
    # Level complete when progress bar (row 63) is full or some condition is met.
    # In observed data, it's not yet complete.
    # Usually, completion is when all targets are hit or progress bar is full.
    return np.all(grid[63] == 5)