import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION1: Up, ACTION2: Down, ACTION3: Left, ACTION4: Right
    # Based on the delta changes, it seems like a "block" of colors moves or shifts.
    # The game involves moving a specific set of pixels (the player/object) and potentially interacting with other elements.
    # In thiss case, the same 5x5 area (roughly) is shifted by 1 unit in each direction.
    # laout suggests therethought be some 'walls' or 'boundary' cells (color 4).
    #
    # Let's identify the object being moved. Looking at the deltas, the object consists of 
    # of rows 25-29, 40-44, etc., and the} {// This is part of<|channel>thought
    # Actually, looking at the deltas, the object is a block of size 5x5 starting at r25c19.
    # Object properties:
    # Rows: [25, 26, 27, 28, 29]
    # Cols: [19, 20, 21, 22, 23]
    # The values are 12, 12, 9, 9, 9 for those rows respectively.
    #
    # Let's find the current position of the "player" object.
    # The player object is a unique set of pixels that move together.
    # In the initial grid, the same pattern appears multiple times.
    # But only one moves.
    # The player object is likely the block of colors [12, 12, 9, 9, 9] in columns 19-23 (or shifted).
    #
    # Based on the transitions:
    # ACTION1: Up (-1 row)
    # ACTION2: Down (+1 row)
    # ACTION3: Left (-1 col)
    # ACTION4: Right (+1 col)
    #
    # Action 3 (Left): r45c24 -> r45c19 (shift left by 5)
    # Action 4 (Right): r25c19 -> r25c24 (shift right by 5)
    # Wait, the shift is by 5 units? No, looking at the deltas:
    # ACTION3: r45c24:12x5,3x5 ... r61c14:3x1. This means cells at c24 became 12 and c29 became 3.
    # It looks like a "swap" or "move".
    # Let's simplify: it's a grid movement game. The player object moves.
    #
    # Identify the moving block:
    # In INITIAL GRID, there are blocks of colors [12, 12, 9, 9, 9] in rows 45-49, cols 25-29.
    # And blocks of colors [12, 12, 9, 9, 9] in rows 25-29, cols 20-24.
    # Actually, let's just track the pixels that change.
    #
    # Looking at the transitions again:
    # ACTION3: shifted something from col 24 to col 19. (Delta -5)
    # ACTION4: shifted something from col 19 to col 24. (Delta +5)
    # ACTION1: shifted something from row 40 to row 35. (Delta -5)
    #
    # This is very strange. The shift is exactly 5 units.
    # Let's look at the coordinates: r45c24 -> r45c19. That's a move of 5 columns.
    # r40c19 -> r35c19. That's a move of 5 rows.
    #
    # It seems the "player" moves in increments of 5.
    # Let's find the player object: it' same pattern of values.
    # Pattern: Row 0: 12x5, Row 1: 12x5, Row 2: 9x5, Row 3: 9x5, Row 4: 9x5.
    #
    # Find this block in the grid. Move it by 5 pixels in the given direction.
    # Replace old position with background (color 3).
    #
    # Directions: ACTION1=Up(-5), ACTION2=Down(+5), ACTION3=Left(-5), ACTION4=Right(+5).

    new_grid = grid.copy()
    
    # Define the player block pattern
    pattern = np.array([
        [12]*5,
        [12]*5,
        [9]*5,
        [9]*5,
        [9]*5
    ])
    
    # Find current top-left corner of the moving block
    # We search for the first occurrence of the pattern
    found = False
    start_r, start_c = -1, -1
    for r in range(64 - 5):
        for c in range(64 - 5):
            if np.array_equal(grid[r:r+5, c:c+5], pattern):
                start_r, start_c = r, c
                found = True
                break
        if found: break
    
    if not found: return new_grid
    
    # Determine movement
    dr, dc = 0, 0
    if action == 1: dr = -5
    elif action == 2: dr = 5
    elif action == 3: dc = -5
    elif action == 4: dc = 5
    else: return new_grid
    
    new_r, new_c = start_r + dr, start_c + dc
    
    # Boundary check
    if not (0 <= new_r <= 64 - 5 and 0 <= new_c <= 64 - 5):
        return new_grid
    
    # Move block: clear old, set new
    # The background color seems to be 3 based on the deltas (e.g., "3x5")
    new_grid[start_r:start_r+5, start_c:start_c+5] = 3
    new_grid[new_r:new_r+5, new_c:new_c+5] = pattern
    
    return new_grid

def is_level_complete(grid):
    # No win state provided in observed transitions, but usually it's reaching a target.
    # Since we don't have one, return False unless a specific condition is met.
    return False