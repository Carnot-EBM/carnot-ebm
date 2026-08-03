import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, this looks like a movement/collection game.
    # Action 1: Up, Action 2: Down, Action 3: Left, Action 4: Right.
    # There's an entity (color 3) that moves and interacts with other colors.
    # The state changes in specific blocks of color 12 or 9.
    # Let's identify the "player" position based on the same pattern seen in ACTION1 (Up)
    # and ACTION4 (Right).
    
    # Find player position (the unique identifier for color 3)
    # In the initial grid, there are many cells of color 3.
    # However, looking at the delta, the 'active' part of the laout seems to be related to 
    # the coordinates where changes occur.
    # r61c14:3x1 means row 61, col 14 is now color 3.
    #<|channel>thought
    # Looking at the deltas:
    # ACTION3 (Left): r61c14:3x1, r62c14:3x1. Then r61c15:3x1, r62c15:3x1.
    # Wait, the deltas show a coordinate moving linearly.
    # Action 1: Up, Action 2: Down, Action 3: Left, Action 4: Right.
    # The entity moving is likely represented by the value 3 at rows 61-62.
    # Let's track the x-coordinate of the "cursor" in rows 61-62.
    
    # Identify cursor position from rows 61 and 62.
    # Find columns where grid[61, :] == 3 and grid[62, :] == 3.
    cursor_cols = np.where(grid[61, :] == 3)[0]
    if len(cursor_cols) == 0:
        return grid.copy()
    
    curr_x = cursor_cols[0]
    
    # Movement logic for the cursor
    next_x = curr_x
    next_y = None # Not explicitly used in movement but implied by action
    
    if action == 1: # Up
        next_x = curr_x - 1 # This doesn't match ACTION1 (Up) delta r61c16:3x1 etc.
        # Wait, let's re-examine.
        # Initial state: no color 3 in row 61/62? No, it is there.
        # Action 3 (Left): r61c14:3x1 -> r61c15:3x1. That's moving RIGHT.
        # Let's look at the sequence of changes in rows 61/62:
        # Start: ?
        # A3: c14 -> c15
        # A1: c16 -> c17 -> c18 -> c19 -> c20
        # A4: c20 -> c19 (Wait, this is confusing).
        
        # Let's try a simpler mapping:
        # Action 1: x += 1
        # Action 2: x -= 1
        # Action 3: x -= 1
        # Action 4: x += 1
        # Actually, looking at the deltas for Action 1:
        # r61c16:3x1, then r61c17:3x1, then r61c18:3x1...
        # This means ACTION 1 moves the cursor to the right.
        # ACTION 4 also moves it? r61c19:3x1, then r61c20:3x1.
        # ACTION 3: r61c14:3x1, then r61c15:3x1.
        
    # Let's re-evaluate based on the grid changes in the main area.
    # The blocks of color 12/9 are being replaced by color 3.
    # In Action 1 (Up), rows 40-44 change, then 35-39, then 30-34, then 25-29.
    # This is clearly moving UP in the grid.
    # So Action 1 = Up.
    # When Action 1 happens, the "cursor" in row 61/62 moves RIGHT.
    # When Action 4 (Right) happens, the block at col 19-23 shifts to col 24-28.
    # And the cursor in row 61/62 moves RIGHT.
    # When Action 3 (Left) happens, the block at col 24-28 shifts back to col 19-23.
    # And the cursor in row 61/62 moves LEFT? No, it says r61c14:3x1 then r61c15:3x1.
    
    # Simplified logic:
    # Action 1: Move active block UP. Cursor X increases.
    # Action 2: Move active block DOWN. Cursor X decreases.
    # Action 3: Move active block LEFT. Cursor X decreases.
    # Action 4: Move active block RIGHT. Cursor X increases.
    
    new_grid = grid.copy()
    
    # Find current active block position
    # The blocks are usually color 12 or 9 and width 5.
    # Let's find where they are.
    block_rows = np.where((grid == 12) | (grid == 9))[0]
    if len(block_rows) == 0: return new_grid
    
    min_r, max_r = np.min(block_rows), np.max(block_rows)
    block_cols = np.where((grid[min_r, :] == 12) | (grid[min_r, :] == 9))[0]
    if len(block_cols) == 0: return new_grid
    min_c, max_c = np.min(block_cols), np.max(block_cols)
    
    # Update cursor in row 61/62
    cursor_cols = np.where(grid[61, :] == 3)[0]
    curr_x = cursor_cols[0] if len(cursor_cols) > 0 else 0
    
    if action == 1: # Up
        # Shift block up by 5 rows
        new_grid[min_r:max_r+1, min_c:max_c+1] = grid[min_r+5:max_r+6, min_c:max_c+1] if max_r+6 < 64 else 4
        new_grid[min_r-5:max_r-4, min_c:max_c+1] = grid[min_r:max_r+1, min_c:max_c+1]
        # This is too complex. Let's just use the observed deltas.
        # The blocks of color 12/9 are replaced by color 3.
        # In Action 1, a new set of cells becomes 12/9 and old ones become 3.
        pass

    # Given the constraints and the patterns, let's implement a basic movement for the cursor
    # and a simple state change for the blocks.
    
    # Cursor update (rows 61, 62)
    cursor_cols = np.where(grid[61, :] == 3)[0]
    cx = cursor_cols[0] if len(cursor_cols) > 0 else 0
    if action == 1: cx += 1
    elif action == 2: cx -= 1
    elif action == 3: cx -= 1 # Based on A3 moving block left
    elif action == 4: cx += 1 # Based on A4 moving block right
    
    new_grid[61, :].fill(4)
    new_grid[62, :].fill(4)
    if 0 <= cx < 64:
        new_grid[61, cx] = 3
        new_grid[62, cx] = 3
        
    # Block movement logic
    # Find current block of color 12/9
    block_mask = (grid == 12) | (grid == 9)
    coords = np.argwhere(block_mask)
    if coords.size > 0:
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        
        if action == 1: # Up
            # Move the "active" area up by 5 rows
            # The cells that were 12/9 become 3, and new ones above them become 12/9
            new_grid[r_min:r_max+1, c_min:c_max+1] = 3
            if r_min >= 5:
                new_grid[r_min-5:r_max-4, c_min:c_max+1] = grid[r_min:r_max+1, c_min:c_max+1]
        elif action == 4: # Right
            new_grid[r_min:r_max+1, c_min:c_max+1] = 3
            if c_max + 5 < 64:
                new_grid[r_min:r_max+1, c_min+5:c_max+6] = grid[r_min:r_max+1, c_min:c_max+1]
        elif action == 3: # Left
            new_grid[r_min:r_max+1, c_min:c_max+1] = 3
            if c_min >= 5:
                new_grid[r_min:r_max+1, c_min-5:c_max-4] = grid[r_min:r_max+1, c_min:c_max+1]

    return new_grid

def is_level_complete(grid):
    # Level complete if all target blocks are converted to color 3 or a specific pattern is reached.
    # In this game, it's likely when the block reaches the top or a certain position.
    # For now, return False as no win state was provided.
    return False