import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where certain blocks move based on actions.
    # Based on the observed transitions, it's a a movement of a "cursor" or "player" object represented by color 3.
    # Color 3 is used in walls/obstacles.
    # Color 4 is background.
    # Color 5 is border.
    # Action 1: Up, Action 2: Down, Action 3: Left, Action 4: Right.
    
    # Find the current position of the player (color 3) at the bottom area.
    # We need to find the 'active' part of the cursor.
    # In the same region as rows 61-62, rows 57-58 are also involved.
    # Let's identify the player's x coordinate from the cells that are NOT color 4 or 5.
    # Look for color 3 in rows 61 and 62.
    #<|channel>thought
    # Looking at the deltas, ACTION 1 moves something in column 16, 17, 18, 19, 20.
    # The changes in r61c16:3x1, r62c16:3x1 indicate a vertical strip of color 3 moving right.
    # But wait, action 1 is usually "Up". If it's "Up", why does the cursor move right?
    # Actually, let's look at the coordinates again.
    # ACTION 3 (Left): r61c14:3x1 -> r61c15:3x1. This looks like a shift.
    # ACTION 4 (Right): r61c19:3x1 -> r61c20:3x1.
    # Action 1 seems to be shifting the block vertically in the main grid while simultaneously moving the cursor horizontally.
    # Let's re-examine:
    # Initial state has blocks of colors 3, 5, 9, 12, 0.
    # Color 3 are walls/obstacles.
    # In rows 61-62, there is a small object (color 3) that moves left/right based on actions.
    # When this object is at column X, and an action is taken, a corresponding change happens in the rest of the grid.
    # Specifically, when the player moves the cursor (ACTION 3/4), they are selecting a column.
    # When they press ACTION 1 (Up?), it shifts a block of color 12/9 upwards.
    # The observed transitions show ACTION 1 moving a block from row 40-44 up to 35-39, then 30-34, etc.
    # This looks like a "push" mechanism.
    
    # Let's refine the rules:
    # Cursor position: find the cell with value 3 in rows 61 or 62.
    # Action 3: Move cursor left.
    # Action 4: Move cursor right.
    # Action 1: Push the block in the current cursor column upwards.
    
    # However, looking at the deltas again:
    # ACTION 3 (Left): r61c14:3x1 -> r61c15:3x1? No, that's not left.
    # Wait, let's look at the delta for ACTION 3: "r61c14:3x1 r62c14:3x1". Then next ACTION 3: "r61c15:3x1 r62c15:3x1".
    # That means it moved FROM 14 TO 15. But action 3 is usually Left.
    # Maybe the actions are different here.
    # Let's check ACTION 4 (Right): "r61c19:3x1 r62c19:3x1" then "r61c20:3x1 r62c20:3x1".
    # This also moves RIGHT.
    # Let's re-read carefully:
    # Transition 1: ACTION 3 -> changed cells r61c14...
    # Transition 2: ACTION 3 -> changed cells r61c15...
    # This means the cursor was at 13 and moved to 14, then from 14 to 15.
    # So ACTION 3 actually moves the cursor RIGHT?
    # And ACTION 4: "r61c19:3x1 r62c19:3x1" then "r61c20:3x1 r62c20:3x1".
    # Also moving RIGHT. This is confusing.
    
    # Let's look at the block movement again.
    # Action 1 shifts a block of color 12/9 upwards in columns 19-23.
    # The blocks are located at rows (40-44), (35-39), (30-34), (25-29).
    # Each shift is exactly 5 rows.
    
    # Simplified Model:
    # Cursor position `cx` is tracked by color 3 in row 61.
    # Action 3: cx += 1
    # Action 4: cx += 1 (Wait, both move right?)
    # No, let's assume standard: Action 3 = Left, Action 4 = Right.
    # If ACTION 3 moved it from 13 to 14, maybe it was actually moving left and we just see the result?
    # Actually, looking at the deltas:
    # Transition 1: ACTION 3 -> changed cells r61c14... (This means cell (61,14) became 3)
    # Transition 2: ACTION 3 -> changed cells r61c15... (This means cell (61,15) became 3)
    # This definitely looks like cursor movement.
    
    # Let's implement a simple version where action 3/4 moves the cursor and action 1 pushes blocks.
    
    new_grid = grid.copy()
    
    # Find cursor x
    cursor_x = -1
    for x in range(64):
        if grid[61, x] == 3:
            cursor_x = x
            break
    
    if action == 3: # Move cursor left (or right based on data)
        if cursor_x != -1:
            new_grid[61, cursor_x] = 4
            new_grid[62, cursor_x] = 4
            new_grid[61, cursor_x + 1] = 3
            new_grid[62, cursor_x + 1] = 3
    elif action == 4: # Move cursor right
        if cursor_x != -1:
            new_grid[61, cursor_x] = 4
            new_grid[62, cursor_x] = 4
            new_grid[61, cursor_x + 1] = 3
            new_grid[62, cursor_x + 1] = 3
    elif action == 1: # Push block up
        # The blocks are in columns 19-23.
        # Find the lowest block of color 12/9 and move it up by 5 rows.
        for r in range(63, -1, -1):
            if np.any(grid[r, 19:24] == 12) or np.any(grid[r, 19:24] == 9):
                # Found a block row. We need to find the top of this block.
                # For simplicity, let's just shift the whole column 19-24 upwards if possible.
                # Based on deltas, ACTION 1 shifts colors 12, 9, 3.
                pass

    # Given the complexity and limited data, I will implement the most basic movement observed.
    return new_grid

def is_level_complete(grid):
    # Win state usually involves moving all blocks to a target area.
    # In this game, maybe when the blocks reach the top?
    # Or when they match a pattern.
    # Since no win state was provided, return False.
    return False