import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action 3 corresponds to a leftward movement of some object/entity.
    # Based on the observed transitions, it's not entirely clear what entity is moving.
    # However, we can see that specific patterns are moving left.
    # The pattern in rows 19 and 20 consists of [9, 10, 2, 2].
    # This is being shifted left by 2 columns each time.
    # We can actually observe that own values are changing at r19c37, then c35, c33, c31...
    # In ACTION3, cells in rows 19 and 20 are updated.
    #<|channel>thought
    # Looking closer at the observed transitions for ACTION3:
    # Transition 1: r19c37:9x1,10x1,2x2; r20c37:9x1,10x1,2x2
    # Transition 2: r6c62:0x2; r7c62:0x2; r19c35:9x1,10x1,2x2; r20c35:9x1,10x1,2x2
    # Transition 3: r8c62:0x2; r9c62:0x2; r19c33:9x1,10x1,2x2; r20c33:9x1,10x1,2x2
    # Transition 4: r19c31:9x1,10x1,2x2; r20c31:9x1,10x1,2x2
    # Transition 5: r10c62:0x2; r11c62:0x2; r19c29:9x1,10x1,2x2; r20c29:9x1,10x1,2x2
    # Transition 6: r19c27:9x1,10x1,2x2; r20c27:9x1,10x1,2x2
    # Transition s: ... and so on.
    # It seems ACTION3 moves a specific pattern [9, 10, 2, 2] in rows 19-20 leftward by 2 units.
    # Additionally, some cells at the right edge (col 62, 63) are being set to 0.
    # The "object" is likely the block of colors starting with 9, 10...
    # Let's implement this movement logic.

    if action == 3:
        # Find the object pattern [9, 10, 2, 2] in rows 19 and 20.
        # We need to find where it currently starts.
        new_grid = grid.copy()
        
        # Search for the start of the pattern [9, 10, 2, 2] in row 19.
        start_col = -1
        for c in range(64):
            if grid[19, c] == 9 and c+3 < 64 and \
               grid[19, c+1] == 10 and grid[19, c+2] == 2 and grid[19, c+3] == 2:
                start_col = c
                break
        
        if start_col != -1:
            # Move it left by 2 units.
            target_col = max(0, start_col - 2)
            
            # Clear old position (set back to background color 5).
            # Based on observed data, we don't see explicit clearing, but usually objects move.
            # However, looking at the delta, only the NEW positions are listed.
            # Let's assume a simple shift.
            
            # The delta shows r19c37:9x1,10x1,2x2 which means cells at 37, 38, 39, 40 become [9, 10, 2, 2].
            # Then r19c35... then r19c33...
            # This suggests the pattern is being "stamped" or moved.
            
            # To match the behavior exactly:
            # 1. Find current pos of [9, 10, 2, 2] in row 19/20.
            # 2. Shift it left by 2.
            # 3. Set new values.
            # 4. Set old values to 5? No, let' same as initial grid logic.
            
            pattern = [9, 10, 2, 2]
            for r in [19, 20]:
                # Clear previous (this is an assumption since deltas only show changes)
                # In many ARC games, moving objects leave background behind.
                # Let's try to find all instances of this pattern and move them.
                for c in range(64):
                    if grid[r, c] == 9 and c+3 < 64 and \
                       grid[r, c+1] == 10 and grid[r, c+2] == 2 and grid[r, c+3] == 2:
                        new_grid[r, c:c+4] = 5 # Background
            
            for r in [19, 20]:
                # Find current start again on original grid
                for c in range(64):
                    if grid[r, c] == 9 and c+3 < 64 and \
                       grid[r, c+1] == 10 and grid[r, c+2] == 2 and grid[r, c+3] == 2:
                        target_col = max(0, c - 2)
                        new_grid[r, target_col:target_col+4] = pattern
                        break

        # Handle the right-edge cells changing to 0 (observed in some transitions).
        # Transition 2: r6c62:0x2; r7c62:0x2
        # Transition 3: r8c62:0x2; r9c62:0x2
        # Transition 5: r10c62:0x2; r11c62:0x2
        # This looks like a sequence. Let's track it with a simple counter or state.
        # Since engine must be pure, we can't have a persistent counter.
        # Maybe it depends on the current position of the object?
        # Object at c=37 -> no change to edge.
        # Object at c=35 -> r6, r7 become 0.
        # Object at c=33 -> r8, r9 become 0.
        # Object at c=31 -> no change.
        # Object at c=29 -> r10, r11 become 0.
        # It seems every 2nd move left triggers two rows at the edge to turn 0.
        # Move 1 (to 37): -
        # Move 2 (to 35): r6, r7
        # Move 3 (to 33): r8, r9
        # Move 4 (to 31): -
        # Move 5 (to 29): r10, r11
        # This is slightly irregular. Let's look at the start_col again.
        # Initial: pattern was likely at c=39? (since first action moves it to 37).
        # If we can find a relationship between start_col and which rows are 0...
        # start_col 35 -> rows 6, 7
        # start_col 33 -> rows 8, 9
        # start_col 29 -> rows 10, 11
        # The formula could be row = (some constant) - start_col.
        # For 35: 6 = K - 35 => K = 41. Row 6, 7.
        # For 33: 8 = K - 33 => K = 41. Row 8, 9.
        # For 29: 10 = K - 29 => K = 39. Wait.
        # Let's re-examine:
        # Transition 2: Object moves to 35, Rows 6, 7 change.
        # Transition 3: Object moves to 33, Rows 8, 9 change.
        # Transition 5: Object moves to 29, Rows 10, 11 change.
        # It seems for every 2 units the object moves left, two rows at col 62,63 are cleared?
        # No, it's not every move. Move 1(37), 2(35), 3(33), 4(31), 5(29).
        # Only transitions 2, 3, 5 have edge changes.
        # This is too specific. I will focus on the movement of the pattern.

        return new_grid

    return grid

def is_level_complete(grid):
    # A level is usually complete when a goal state is reached.
    # In this case, we don't have a win state grid, but typically it involves
    # clearing all target cells or moving an object to a destination.
    # Given the data, let's assume completion if the right-edge (col 62, 63)
    # has many zeros or the object reaches a certain column.
    # Without more info, return False unless a clear condition is met.
    return False