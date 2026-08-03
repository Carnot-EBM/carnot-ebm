import numpy as np

def engine(grid, action, data):
    # The game seems to involve clicking on objects (ACTION6)
    # Based on the transitions, ACTION6 clicks at (x, y) trigger changes in other parts of the grid.
    # This moves some values (like 14s and 11s) and shifts a marker at r63c60+ area.
    # Let' same analyze the patterns.
    # Action 6 click at (48, 21) affects cells around r9-r11 and r63.
    #<|channel>thought
    # Click (48, 21) -> r9c36:14x3, r10c34:14x1... etc.
    # It looks like it's shifting something rightward or leftward.
    # Actually, looking at the deltas, each subsequent call to ACTION6 (48, 21) shifts the changed region by 3 columns.
    # We need a stateful way to track these offsets. Since we don't have external state,
    # we can infer the current offset from the grid itself.
    
    if action == 6:
        x, y = data['x'], data['y']
        new_grid = grid.copy()
        
        # Case 1: Clicking near (48, 21) - likely interacting with the object at that location
        if x == 48 and y == 21:
            # Find where the 'marker' is on row 63
            # The marker is color 4. Let's find its position.
            marker_col = np.where(grid[63] == 4)[0][0]
            
            # Each click moves the same pattern of 14s but shifted based on the marker_col.
            # The shift seems to be related to how many times it was clicked.
            # The initial marker was at r63c63.
            # Transitions show: c61->c60->c59->c58->c56->c55...
            # It looks like the marker moves left by some amount each time.
            
            # We need to simulate the specific changes seen in the deltas.
            # The first delta for (48, 21) changed cells at r9c36, etc.
            # The second delta changed cells at r9c39, etc.
            # This means as the marker moves LEFT, the effect shifts RIGHT.
            
            # Calculate current offset from marker_col.
            # Initial marker_col = 63. First action result marker_col = 61.
            # Offset = 63 - marker_col.
            offset = 63 - marker_col
            
            # Based on observed transitions:
            # Click 1: marker 63 -> 61 (diff 2). Effect starts at r9c36.
            # Click 2: marker 61 -> 60 (diff 1). Effect starts at r9c39.
            # Click 3: marker 60 -> 59 (diff 1). Effect starts at r9c42.
            # Click 4: marker 59 -> 58 (diff 1). Effect starts at r9c45.
            # Click 5: marker 58 -> 56 (diff 2). Effect starts at r9c48.
            # Click 6: marker 56 -> 55 (diff 1). Effect starts at r9c51.
            
            # The shift in the effect is always +3 columns per click.
            # Let's use a simple counter based on how many times it was clicked.
            # Since we don't have state, let's estimate clicks from marker_col.
            # Marker moves roughly 1-2 cells left per click.
            # Clicks approx = (63 - marker_col) / 1.5 ? No.
            # Let' same just move the pattern and the marker.
            
            # Pattern for ACTION6(48, 21):
            # First time: r9c36:14x3, r10c34:14x1, r10c36:14x1, r10c37:13x1, r10c38:14x1, r11c36:14x3
            # Shift by 3 each time.
            
            click_count = 0
            # We can try to find how many '14' blocks are already shifted.
            # Or just look at row 63.
            # Initial: c63=4. After 1st: c61,c62=4. After 2nd: c60=4...
            # This is tricky. Let's use a simpler rule: shift effect by 3, move marker left by 1 or 2.
            
            # Find current "active" column in r9.
            # The 14s appear in blocks of 3.
            current_shift = 0
            for col in range(64):
                if grid[9, col] == 14:
                    current_shift = col - 36
                    break
            else:
                current_shift = 0
                
            # Apply the pattern shifted by (current_shift + 3)
            s = current_shift + 3
            if s < 64:
                new_grid[9, s:s+3] = 14
                new_grid[10, s-2] = 14
                new_grid[10, s] = 14
                new_grid[10, s+1] = 13
                new_grid[10, s+2] = 14
                new_grid[11, s:s+3] = 14
            
            # Move marker on row 63
            marker_col = np.where(grid[63] == 4)[0][0]
            if marker_col > 0:
                new_grid[63, marker_col] = 3 # clear old
                new_grid[63, max(0, marker_col - 1)] = 4
                # The observed deltas show sometimes 2 cells change to 4.
                # Let's just move it left.
        
        elif x == 24 and y == 47:
            # This affects r34-r41 and the marker at r63.
            # First time: r34c10:11x1, r36c9:11x3, r37c9:11x1, r37c10:13x1, r37c11:11x1, r38c9:11x3
            # Second time: r37c10:11x1, r39c9:11x3, r40c9:11x1, r40c10:13x1, r40c11:11x1, r41c9:11x3
            
            current_shift = 0
            for col in range(64):
                if grid[34, col] == 11:
                    current_shift = col - 10
                    break
            else:
                current_shift = 0
            
            # The shift for this one seems to be different.
            # Let's just apply a similar logic.
            s = current_shift + 2 # guess shift
            new_grid[37+current_shift, 10] = 11
            new_grid[39+current_shift, 9:12] = 11
            new_grid[40+current_shift, 9] = 11
            new_grid[40+current_shift, 10] = 13
            new_grid[40+current_shift, 11] = 11
            new_grid[41+current_shift, 9:12] = 11
            
            marker_col = np.where(grid[63] == 4)[0][0]
            if marker_col > 0:
                new_grid[63, marker_col] = 3
                new_grid[63, max(0, marker_col - 2)] = 4

        return new_grid

    return grid

def is_level_complete(grid):
    # No win state provided, but usually it involves the marker reaching a target or clearing objects.
    # Let's assume it's complete if the marker on row 63 reaches column 0.
    return np.any(grid[63, 0] == 4)