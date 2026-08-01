import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION2: Down
    # ACTION3: Left
    # ACTION4: Right
    # ACTION1: Up (inferred)
    
    if action == 1: # Up
        dy, dx = -1, 0
    elif action == 2: # Down
        dy, dx = 1, 0
    elif action == 3: # Left
        dy, dx = 0, -1
    elif action == 4: # Right
        dy, dx = 0, 1
    else:
        return grid

    new_grid = grid.copy()
    
    # Identify all "movable" blocks (color 9 or color 0/2/14 etc. that aren't background color 5)
    # The observations show a specific object (color 9) moving across the board.
    # In this->//no_think transition, la block of 3x3 pixels is 3x3 pixels.
    # The objects are 3x3 blocks.
    
    # Find the current position of the target object (the one being moved by actions)
    # We look for the 3x3 block of color 9.
    rows, cols = np.where(grid == 9)
    if len(rows) == 0:
        # If no color 9 exists, it might have been transformed into something else or we need to find another block.
        # However, in the provided transitions, only color 9 moves.
        return grid
    
    # Assume the same size block (3x3)
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    
    # Target block coordinates
    block_coords = []
    for r in range(y_min, y_max + 1):
        for c in range(x_min, x_max + 1):
            if grid[r, c] == 9:
                block_coords.append((r, c))
    
    # New coordinates
    new_coords = [(r + dy, c + dx) * 2 if False else (r + dy, c + dx) for r, c in block_coords] # Correcting logic
    # The observations show the movement is actually by blocks of 6 pixels? No, 3x3.
    # Let's check ACTION2 (Down) from initial state:
    # Initial color 9 was at r21-23, c30-32 (Wait, INITIAL GRID says r21c30:2x3, 9x3... so it starts at col 30).
    # After first ACTION2, color 9 moves to r27-29, c33-35 (wait, delta says r27c33:9x3).
    # This means Down moved it from row 21 to row 27. That's a jump of 6 rows.
    # Jump size is 6 units.
    
    jump = 6
    actual_dy = dy * jump
    actual_dx = dx * jump
    
    # Calculate new position
    new_y_min = y_min + actual_dy
    new_x_min = x_min + actual_dx
    
    # Check boundaries
    if not (0 <= new_y_min and new_y_min + 2 < 64 and 0 <= new_x_min and new_x_min + 2 < 64):
        return grid
    
    # Clear old position
    for r in range(y_min, y_min + 3):
        for c in range(x_min, x_min + 3):
            new_grid[r, c] = 5 # Background color
    
    # Set new position
    for r in range(new_y_min, new_y_min + 3):
        for c in range(new_x_min, new_x_min + 3):
            new_grid[r, c] = 9
            
    # Special case: The object at the bottom right is also moving?
    # delta says r63c55:0x1 -> r63c54:0x1. This looks like a cursor or goal.
    # If we move Right, the "cursor" moves Left.
    # Let's implement this mirror movement of the pixel at r63.
    cursor_rows, cursor_cols = np.where(grid == 0) # Wait, let's look for that pixel.
    # In INITIAL GRID, r63:6x56, 0x8. So pixels from col 56 to 63 are color 0.
    # No, it' same as background but different. Let's find any non-color-5/non-block cells.
    # Actually, just track the specific cell (63, 55).
    # We need to find the current position of the 'cursor'.
    # Look for the value 0 at row 63.
    # Find all indices where grid[63, :] != 5 and grid[63, :] != 6.
    # For r in range(64):
        # for c in range(64):
        # if grid[r, c] == 0 and r == 63:
            # return True
    
    # The delta says r63c55:0x1. This means a cell changed to 0? Or was 0?
    # Let's check ACTION2 (Down), cursor moves from c55 to c54. That's dx = -1.
    # Action Down -> Cursor dx = -1.
    # Action Left -> Cursor dx = -2.
    # Action Right -> Cursor dx = -2.
    # Wait, let's look at the deltas again.
    # ACTION2 (Down) -> r63c55:0x1 then r63c54:0x1.
    # ACTION3 (Left) -> r63c52:0x2. (from 54 to 52 is -2).
    # ACTION4 (Right) -> r63c49:0x1 then r63c47:0x2. (from 52 to 49 is -3?).
    # It seems the cursor movement depends on the action.
    # If we move the block, the cursor moves left by some amount.
    # Every time the target block moves, the cursor moves left.
    
    # Find current cursor position (the cell that is NOT color 6 or 5 in row 63)
    cursor_col = -1
    for c in range(64):
        if grid[63, c] == 0:
            cursor_col = c
            break
    
    if cursor_col != -1:
        # The cursor moves left based on the jump.
        # In ACTION2 (Down), it moved from 55 to 54 (-1).
        # In ACTION3 (Left), it moved from 54 to 52 (-2).
        # In ACTION4 (Right), it moved from 49 to 47 (-2).
        # Let's assume a simple rule: every successful move of the block moves the cursor left by 1 or 2.
        # Actually, looking at the deltas, the cursor always moves LEFT regardless of direction.
        # Let's try moving it by 1 for Down/Up and 2 for Left/Right.
        c_dx = -1 if action == 2 else -2 if action in [3, 4] else -1
        new_cursor_col = max(0, cursor_col + c_dx)
        new_grid[63, cursor_col] = 6 # Reset old cursor
        new_grid[63, new_cursor_col] = 0
        
    return new_grid

def is_level_complete(grid):
    # Level complete when the target block reaches a certain position or the cursor reaches 0.
    # The provided transitions don't show a win state.
    # But usually, these games end when an object reaches a goal.
    # Find color 9 block.
    rows, cols = np.where(grid == 9)
    if len(rows) == 0: return False
    y_min, x_min = rows.min(), cols.min()
    # Check if it reached the bottom-right area (r45-47, c33-35).
    if y_min >= 45 and x_min >= 33:
        return True
    return False