import numpy as np

def engine(grid, action, data):
    """
    Induces an executable world model for ARC-AGI game 'ls20'.
    Based on observations:
    Action 1 (UP), Action 3 (LEFT), Action 4 (RIGHT) move a set of colored blocks.
    The grid contains static boundaries (color 4, 5) and movable objects (colors 3, 9, 12, 0).
    There's also a cursor/indicator at the bottom (rows 61-62) that moves horizontally.
    """
    new_grid = grid.copy()
    h, w = grid.shape

    # Identify the "cursor" position based on rows 61-62 where color 3 appears
    # The transitions show ACTION3/4 moving the cursor in r61c14 -> c15 and ACTION1 moving it c16 -> c17 etc.
    # This suggests the actions are controlling a player or object.
    
    # Let's look at the block movement patterns:
    # Action 3 (Left): Blocks shift left. e.g., r45c24:12x5 becomes r45c19:12x5. Shift is -5 columns.
    # Action 4 (Right): Blocks shift right. e.g., r25c19:12x5 becomes r25c24:12x5. Shift is +5 columns.
    # Action 1 (Up): Blocks shift up. e.g., r40c19:12x5 shifts to r35c19:12x5. Shift is -5 rows.
    
    shift_val = 5
    
    if action == 3: # LEFT
        # Move blocks of colors {0, 3, 9, 12} left by shift_val if not blocked by color 5/4 boundaries
        # In this specific level, we see a vertical strip of objects shifting.
        for r in range(h):
            for c in range(w - 1, -1, -1):
                val = grid[r, c]
                if val in [0, 3, 9, 12]:
                    target_c = c - shift_val
                    if target_c >= 0 and grid[r, target_c] == 4: # Only move into 'empty' space (color 4)
                        new_grid[r, target_c] = val
                        new_grid[r, c] = 4
        # Update cursor
        for r in [61, 62]:
            cursor_cols = np.where(grid[r] == 3)[0]
            if len(cursor_cols) > 0:
                curr_c = cursor_cols[0]
                if curr_c - 1 >= 0:
                    new_grid[r, curr_c] = 4
                    new_grid[r, curr_c - 1] = 3

    elif action == 4: # RIGHT
        for r in range(h):
            for c in range(w):
                val = grid[r, c]
                if val in [0, 3, 9, 12]:
                    target_c = c + shift_val
                    if target_c < w and grid[r, target_c] == 4:
                        new_grid[r, target_c] = val
                        new_grid[r, c] = 4
        # Update cursor
        for r in [61, 62]:
            cursor_cols = np.where(grid[r] == 3)[0]
            if len(cursor_cols) > 0:
                curr_c = cursor_cols[0]
                if curr_c + 1 < w:
                    new_grid[r, curr_c] = 4
                    new_grid[r, curr_c + 1] = 3

    elif action == 1: # UP
        for r in range(h):
            for c in range(w):
                val = grid[r, c]
                if val in [0, 3, 9, 12]:
                    target_r = r - shift_val
                    if target_r >= 0 and grid[target_r, c] == 4:
                        new_grid[target_r, c] = val
                        new_grid[r, c] = 4
        # Update cursor (Action 1 moves the cursor right in observations!)
        for r in [61, 62]:
            cursor_cols = np.where(grid[r] == 3)[0]
            if len(cursor_cols) > 0:
                curr_c = cursor_cols[0]
                if curr_c + 1 < w:
                    new_grid[r, curr_c] = 4
                    new_grid[r, curr_c + 1] = 3

    return new_grid

def is_level_complete(grid):
    # No win state provided in observed transitions. 
    # Typically ARC levels are complete when a specific pattern is formed or objects reach a goal.
    # Without a WIN STATE example, we return False.
    return False