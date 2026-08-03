import numpy as np

def engine(grid, action, data):
    """
    Induces the world model for sc25.
    Observation analysis:
    - Grid size: 64x64.
    - Background color: 5.
    - A specific moving object exists at rows 19 and 20 with values [9, 10, 2, 2].
    - ACTION3 causes this object to shift 2 units to the left per step.
    - ACTION3 also triggers a side effect where pairs of cells at col 62, 63 change from 14 to 0.
    - Specifically, looking at the deltas:
        Step 1: r19c37, r20c37 (shift)
        Step 2: r6c62, r7c62 (toggle), r19c35, r20c35 (shift)
        Step 3: r8c62, r9c62 (toggle), r19c33, r20c33 (shift)
        ...and so on.
    - The toggle pattern is: every second ACTION3 call toggles two new rows in columns 62-63.
    """
    new_grid = grid.copy()
    
    if action == 3:
        # 1. Move the object [9, 10, 2, 2] in rows 19 and 20
        # Find current position of the object (looking for color 9)
        for r in [19, 20]:
            coords = np.where(new_grid[r] == 9)[0]
            if len(coords) > 0:
                curr_x = coords[0]
                # Clear old position (assuming it's a 4-wide block)
                # Based on delta "9x1, 10x1, 2x2", length is 4.
                # We need to restore background color 5.
                new_grid[r, curr_x : curr_x + 4] = 5
                
                # Place at new position (shifted left by 2)
                new_x = max(0, curr_x - 2)
                pattern = [9, 10, 2, 2]
                for i, val in enumerate(pattern):
                    if new_x + i < 64:
                        new_grid[r, new_x + i] = val

        # 2. Handle the side effect on columns 62 and 63
        # The observed sequence of toggles:
        # Call 1: None
        # Call 2: r6, r7
        # Call 3: r8, r9
        # Call 4: None
        # Call 5: r10, r11...
        # This suggests a pattern based on total ACTION3 calls or current state.
        # Let's count how many pairs are already 0 in col 62.
        col62 = new_grid[:, 62]
        zeros = np.where(col62 == 0)[0]
        num_toggled_pairs = len(zeros) // 2
        
        # We need to determine if this specific call triggers a toggle.
        # Looking at the logs: Action 3 (0->0), then (0->0)...
        # It seems every other action might trigger it, or there is an internal counter.
        # Since we don't have the counter, we check for the "gap" in the observed deltas.
        # However, simpler logic: find the first row >= 6 that is still color 14 and flip it + next row.
        # But wait, the sequence was: [None], [r6,r7], [r8,r9], [None], [r10,r11]...
        # This looks like: Toggle, Toggle, Skip, Toggle, Toggle, Skip? Or just alternating?
        # Actually, looking closer:
        # Call 1: r19c37 (shift only)
        # Call 2: r6-7c62 (toggle) + shift
        # Call 3: r8-9c62 (toggle) + shift
        # Call 4: r19c31 (shift only)
        # Call 5: r10-11c62 (toggle) + shift
        # Pattern of toggles: No, Yes, Yes, No, Yes, Yes...
        # We can track this by counting how many ACTION3s happened relative to current state.
        # Since engine() is pure, we must derive the 'turn' from the grid.
        # The object position in rows 19/20 is a perfect proxy for turn count.
        # Initial x of color 9 in row 19 is 12+5=17? No, initial grid says r17: 5x12, 9x5... so col 12.
        # Wait, INITIAL GRID r19: 5x12, 9x1, 10x4... so col 12.
        # Current pos / -2 gives us the number of moves.
        curr_pos = np.where(new_grid[19] == 9)[0]
        if len(curr_pos) > 0:
            moves = (12 - curr_pos[0]) // 2
            # Toggle sequence: move 1: no, move 2: yes, move 3: yes, move 4: no, move 5: yes...
            # This is: toggle if (move % 3 != 1) and move > 0?
            # Let's try: if moves % 3 == 2 or moves % 3 == 0:
            # But we are predicting the NEXT state. The next move will be 'moves + 1'.
            next_move = moves + 1
            if next_move > 0 and (next_move % 3 != 1):
                target_row = 6 + (num_toggled_pairs * 2)
                if target_row < 62:
                    new_grid[target_row : target_row + 2, 62:64] = 0

    return new_grid

def is_level_complete(grid):
    """
    The level is likely complete when the object reaches a certain point 
    or all targets in col 62/63 are cleared.
    Given the data, we don't have a win state, but usually it's reaching x=0.
    """
    coords = np.where(grid[19] == 9)[0]
    if len(coords) > 0 and coords[0] <= 2:
        return True
    return False