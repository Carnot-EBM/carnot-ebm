import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION1: Up, ACTION2: Down, ACTION3: Left, ACTION4: Right
    # Based on the deltas, it seems like a "cursor" or "block" of colors moves.
    # The movements are shifts in coordinates.
    # ACTION1: y decreases (Up)
    # ACTION2: y increases (Down)
    # ACTION3: x decreases (Left)
    # ACTION4: x increases (Right)
    
    # Identify the moving object (the 3-color sequence at rows 61-62)
    # Let's find the position of the '3' color cells that move.
    # In INITIAL GRID, they are at r61c14 and r62c14.
    # After first ACTION3, they move to c15? No, wait.
    # Looking at the delta: r61c14:3x1, r62c14:3x1 -> then r61c15:3x1, r62c15:3x1.
    # Wait, the delta is just the NEW values.
    # So if action is ACTION3 (Left), the cursor moves from c14 to c15? That's Right.
    # Action numbers might be different.
    # Let's re-examine:
    # Initial: r61c14, r62c14 = 3.
    # Transition 1 (ACTION3): r61c14 becomes 3, r62c14 becomes 3... no, it says "changed cells".
    # If the cell was already 3, it wouldn't change.
    # The deltas show the new value.
    # ACTION3 (Left) -> r61c14:3x1, r62c14:3x1. This means these cells became 3.
    # Then ACTION3 again -> r61c15:3x1, r62c15:3x1. These cells become 3.
    # This is very confusing. Let's look at the movement of the '3' block in rows 61-62.
    # In INITIAL GRID, they are not explicitly listed as 3.
    # Wait, r61: ... 3x1, 11x41 ...
    # So c14 is 3.
    # First ACTION3: r61c14:3x1, r62c14:3x1. But it was already 3?
    # No, let's check the initial grid carefully.
    # r61: 4x1, 5x10, 4x1, 5x1, 3x1, 11x41...
    # Col indices: 0(4), 1-10(5), 11(4), 12(5), 13(3).
    # So col 13 is value 3.
    # Transition 1 (ACTION3): r61c14:3x1, r62c14:3x1. Now col 14 is 3.
    # Transition 2 (ACTION3): r61c15:3x1, r62c15:3x1. Now col 15 is 3.
    # This means ACTION3 moves the cursor RIGHT.
    # Let's look at ACTION1:
    # Initial cursor at c15.
    # Action 1: r61c16:3x1, r62c16:3x1. Cursor moves to c16.
    # Wait, this is not a simple movement. The '3' block in rows 61-62 is moving.
    # And some other blocks are changing colors.
    # Let's look at the coordinates of the '3' block:
    # Start: c13 -> ACT3 -> c14 -> ACT3 -> c15 -> ACT1 -> c16 -> ACT1 -> c17 -> ACT1 -> c18...
    # It seems like multiple actions move it right?
    # Let's re-read: ACTION1 (Up), ACTION2 (Down), ACTION3 (Left), ACTION4 (Right).
    # If ACTION3 and ACTION1 both move it right, maybe they are just "move" commands.
    # Actually, let's look at the larger grid changes.
    # In ACTION1, we see r40c19:12x5, etc. These are columns 19-23.
    # This looks like a vertical shift of a color block.
    # Block at r45-49, c19-23 shifts up to r40-44, then r35-39, then r30-34, then r25-29.
    # This is clearly ACTION1 = UP.
    # Now let's check ACTION3/ACTION4.
    # ACTION3: r45c24:12x5, ... r46c24:12x5...
    # The block was at c19-23. Now it moves to c24-28.
    # Wait, ACTION3 moved it from c19 to c24? That's RIGHT.
    # Let's re-verify:
    # Initial block: r45-49, c19-23 (approx).
    # ACT3 -> r45c24:12x5. Now it's at c24-28.
    # Then ACT3 again -> r45c19:12x5. Now it's back at c19-23.
    # This looks like a toggle or a shift between two positions.
    # And the cursor in rows 61-62 also moves.
    # Cursor start: c13.
    # ACT3 -> c14.
    # ACT3 -> c15.
    # ACT1 -> c16.
    # ACT1 -> c17.
    # ACT1 -> c18.
    # ACT1 -> c19.
    # ACT4 -> c19 (no change?). No, ACT4 says r61c19:3x1. It was already 19.
    # ACT4 -> c20.
    # So ACTION1 and ACTION3 both move the cursor right? That's weird.
    # Let's look at the block movement again.
    # Block is at (r45-49, c19-23).
    # ACT3: Move to (r45-49, c24-28).
    # ACT3: Move to (r45-49, c19-23).
    # ACT1: Move to (r40-44, c19-23).
    # ACT1: Move to (r35-39, c19-23).
    # ACT1: Move to (r30-34, c19-23).
    # ACT1: Move to (r25-29, c19-23).
    # ACT4: Move to (r25-29, c24-28).
    # ACT4: Move to (r25-29, c19-23) - wait, no, ACT4 moves it from 19 back to 24 then...
    # This is a puzzle where you move a block and a cursor.
    # The cursor in rows 61-62 seems to track the x-coordinate of the block's left edge?
    # Block at c19 -> Cursor at c19.
    # Block at c24 -> Cursor at c24.
    # Let's check:
    # Initial: Block at c19. Cursor at c13.
    # ACT3: Block moves to c24. Cursor moves to c14.
    # ACT3: Block moves to c19. Cursor moves to c15.
    # ACT1: Block moves up. Cursor moves to c16.
    # This doesn't match.
    # Let's try a simpler rule:
    # ACTION1: Shift block UP by 5 rows.
    # ACTION2: Shift block DOWN by 5 rows.
    # ACTION3: Shift block RIGHT by 5 cols.
    # ACTION4: Shift block LEFT by 5 cols.
    # And the cursor (rows 61, 62) just increments its column index every time any action is taken.
    # Let's test this:
    # Start: cursor=13.
    # ACT3: cursor=14. Block moves Right (c19->c24).
    # ACT3: cursor=15. Block moves Left? No, delta says r45c19:12x5. So it moved back to c19.
    # Wait, if ACT3 is "Right", then why did it move back to c19?
    # Maybe ACT3 toggles between c19 and c24?
    # Let's look at the deltas again.
    # Transition 1: ACTION3 -> r45c24... (Block now at c24-28)
    # Transition 2: ACTION3 -> r45c19... (Block now at c19-23)
    # This means ACTION3 toggles the X position.
    # Now let's check ACTION1:
    # Transition 3: ACTION1 -> r40c19... (Block now at r40-44, c19-23)
    # Transition 4: ACTION1 -> r35c19... (Block now at r35-39, c19-23)
    # Transition 5: ACTION1 -> r30c19... (Block now at r30-34, c19-23)
    # Transition 6: ACTION1 -> r25c19... (Block now at r25-29, c19-23)
    # So ACTION1 moves it UP by 5 rows.
    # Now ACTION4:
    # Transition 7: ACTION4 -> r25c19:3x5, 12x5. Wait, this is a complex delta.
    # "r25c19:3x5, 12x5" means from col 19, first 5 cells become 3, next 5 cells become 12.
    # This means the block at c19-23 (value 12/9) is being replaced.
    # The new block is at c24-28? No, "12x5" starts after "3x5".
    # Col 19-23 becomes 3, Col 24-28 becomes 12.
    # So ACTION4 also toggles X position!
    # Let's re-evaluate:
    # ACTION1: Y -= 5
    # ACTION2: Y += 5
    # ACTION3: Toggle X between 19 and 24
    # ACTION4: Toggle X between 19 and 24
    # And cursor in row 61-62 increments column by 1 every action.
    
    # Now we need to implement this. We need to find the current block.
    # The block consists of values {12, 9} and it's 5x5.
    # Let's find its top-left corner (r, c).
    
    res = grid.copy()
    
    # Find the moving block (values 9 or 12)
    block_coords = np.where((grid == 9) | (grid == 12))
    if len(block_coords[0]) == 0:
        return res
    
    min_r = np.min(block_coords[0])
    max_r = np.max(block_coords[0])
    min_c = np.min(block_coords[1])
    max_c = np.max(block_coords[1])
    
    # Current block position
    curr_r = min_r
    curr_c = min_c
    
    # Determine new position
    new_r, new_c = curr_r, curr_c
    if action == 1: # UP
        new_r = max(0, curr_r - 5)
    elif action == 2: # DOWN
        new_r = min(64 - 5, curr_r + 5)
    elif action == 3: # Toggle X
        new_c = 24 if curr_c == 19 else 19
    elif action == 4: # Toggle X
        new_c = 24 if curr_c == 19 else 19
        
    # Update the grid: remove old block, place new block
    # To keep it simple and match deltas, we'll just clear the area and redraw.
    # But wait, the "background" is not uniform. We should only change the moving block's cells.
    # The values in the block are a mix of 9 and 12. Let's preserve that pattern.
    block_pattern = grid[curr_r:curr_r+5, curr_c:curr_c+5].copy()
    
    # Clear old block (set to background color 3 or 4?)
    # Looking at deltas, when the block moves, the old position becomes value 3.
    res[curr_r:curr_r+5, curr_c:curr_c+5] = 3
    
    # Place new block
    res[new_r:new_r+5, new_c:new_c+5] = block_pattern
    
    # Cursor movement in rows 61-62
    cursor_coords = np.where(grid[61:63, :] == 3)
    if len(cursor_coords[0]) > 0:
        curr_cursor_c = cursor_coords[1][0]
        # Move cursor right by 1
        res[61, curr_cursor_c] = 11 # Old cursor becomes something else? No, delta says r61c14:3x1.
        # Let's just find the current '3' and move it.
        # In INITIAL GRID, col 13 is 3. After ACT3, col 14 is 3.
        # So we set the same column to 3 for both rows 61 and 62.
        # But what happens to the previous cursor cell?
        # Delta ACTION3 (level 0->0): changed cells = ... r61c14:3x1 r62c14:3x1
        # This means only the NEW position is mentioned. The old one must have been changed too.
        # Looking at INITIAL GRID: r61 has a 3 at c13.
        # After first action, r61c14 becomes 3. What happened to c13?
        # It probably became 11 (the background color of that area).
        res[61, curr_cursor_c] = 11
        res[62, curr_cursor_c] = 11
        new_cursor_c = min(63, curr_cursor_c + 1)
        res[61, new_cursor_c] = 3
        res[62, new_cursor_c] = 3

    return res

def is_level_complete(grid):
    # No win state provided, but usually it's when the block reaches a target.
    # Let's assume it's complete if the block reaches the top-leftmost possible position.
    block_coords = np.where((grid == 9) | (grid == 12))
    if len(block_coords[0]) == 0:
        return False
    return np.min(block_coords[0]) <= 5 and np.min(block_coords[1]) == 19