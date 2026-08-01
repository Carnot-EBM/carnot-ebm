import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, ACTION4 moves a specific set of cells in the la-la land (the bottom area)
    # and ACTION2/ACTION1 move other elements.
    # The game seems to be about moving blocks of colors.
    
    new_grid = grid.copy()
    
    if action == 4:
        # Action 4 shifts some pattern of pixels from x=15..19 to x=22..26, then 22..26 to 29..33 etc.
        # It's essentially shifting a "cursor" or "active region" by 7 columns.
        # Find current active region based on known patterns in rows 48, 49, 59, 60.
        # Let's look for color 0 in those rows.
        # In r48c15:3x5, it becomes 3x5 (color 3), meaning color 0 is replaced by 3.
        # In r48c22:0x5, it becomes 0x5 (color 0), meaning color 3 is replaced by 0.
        #<|channel>thought
        # Based on observed deltas:
        # Transition 1: r48c15:3x5, r48c22:0x5... (shift cursor from 15-19 to 22-26)
        # Transition 4: r48c22:3x5, r48c29:0x5... (shift cursor from 22-26 to 29-33)
        # Transition 7: r48c29:3x5, r48c36:0x5... (shift cursor from 29-33 to 36-40)
        # This is a shift of +7 columns per ACTION 4.
        
        current_pos = -1
        for x in range(15, 64, 7):
            if np.any(grid[48, x:x+5] == 0):
                current_pos = x
                break
        
        if current_pos != -1:
            next_pos = current_pos + 7
            if next_pos < 64 - 5:
                # Update rows 48, 49, 59, 60 as seen in deltas
                # Row 48 and 60 are blocks of 5 pixels wide
                new_grid[48, current_pos:current_pos+5] = 3
                new_grid[48, next_pos:next_pos+5] = 0
                new_grid[60, current_pos:current_pos+5] = 3
                new_grid[60, next_pos:next_pos+5] = 0
                
                # Row 49 and 59 have specific single pixel updates (e.g., r49c15:3x1, r49c19:3x1)
                # The delta shows r49c15:3x1 and r49c19:3x1 for the first move.
                # This means at pos x, pixels x and x+4 become color 3, while pixels x+7 and x+11 become color 0.
                new_grid[49, current_pos] = 3
                new_grid[49, current_pos+4] = 3
                new_grid[49, next_pos] = 3 # Wait, deltas say r49c22:0x1, r49c26:0x1
                # Let's re-examine: r49c15:3x1, r49c19:3x1 AND r49c22:0x1, r49c26:0x1.
                # So current_pos pixels become 3, next_pos pixels become 0.
                new_grid[49, current_pos] = 3
                new_grid[49, current_pos+4] = 3
                new_grid[49, next_pos] = 0
                new_grid[49, next_pos+4] = 0
                
                new_grid[59, current_pos] = 3
                new_grid[59, current_pos+4] = 3
                new_grid[59, next_pos] = 0
                new_grid[59, next_pos+4] = 0

    elif action == 2:
        # Action 2 shifts some blocks in the bottom area (rows 52-56) and moves a pixel at the very bottom right.
        # The delta shows changes in rows 52, 53, 54, 55, 56.
        # It also moves grid[63, 62] -> [63, 61] etc.
        # Let's implement the specific shift seen in ACTION 2.
        # This looks like it might be shifting objects of color 5 or 7.
        
        # Move the bottom-right marker
        for x in range(64):
            if grid[63, x] == 4:
                new_grid[63, x] = 1 # Or whatever background is, but deltas say r63c62:4x1 then r63c61:4x1
                # Actually, it just says "changed cells", so if r63c62 was 4, and now r63c61 is 4...
                new_grid[63, x] = 1 # Assuming bg is 1 for that row? No, INITIAL GRID says r63:1x63, 4x1 (so col 62 is 4).
                # Wait, r63:1x63, 4x1 means col 0-62 are 1, col 63 is 4.
                # Delta Action 2: r63c62:4x1. That means col 62 becomes 4. Col 63 must become something else.
                # Let's assume a simple shift left of color 4.
                new_grid[63, x] = 1
                if x > 0:
                    new_grid[63, x-1] = 4
                break

    elif action == 1:
        # Action 1 also shifts things in rows 52-56 and moves the marker at [63, x].
        # Similar to ACTION 2 but different blocks.
        for x in range(64):
            if grid[63, x] == 4:
                new_grid[63, x] = 1
                if x > 0:
                    new_grid[63, x-1] = 4
                break

    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it involves reaching a certain configuration.
    # We'll return False as we don't have enough info to determine the win condition.
    return False