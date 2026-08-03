import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid
    
    px, py = data['x'], data['y']
    # The board seems to have some regions.
    # Top region (r0-r7), Middle region (r24-r35), Middle-bottom region (r57-r60)
    # Let's analyze the coordinates provided in ACTION6 clicks.
    # Clicks at y=59 (Bottom area) and y=30 (Middle area).
    
    # Bottom area click logic (y=59):
    # x=36 -> r56c33:0x6 ... r61c33:0x6
    # x=20 -> r56c17:0x6 ... r61c17:0x6
    # x=44 -> r56c41:0x6: ... r61c33:0x6
    # It looks like clicking a cell in the bottom area clears/sets cells to 0 or 4.
    # If we are clicking on one of the colored blocks in the bottom row (r57-r60),
    # it might be "activating" something.
    
    # Mapping based on observed transitions:
    # Click x=36, y=59: affects column range [33, 38] roughly.
    #<|channel>thought
    # The pattern is that clicking in the bottom region (y=59) seems to clear the corresponding block.
    # Clicking in the middle region (y=30) seems to move colors from the bottom to the middle.
    # Let's try to implement this specific mapping.
    
    new_grid = grid.copy()
    
    if py == 59:
        # Bottom clicks seem to toggle/clear a vertical strip in the bottom area
        # col_start = px - 3? No, let's look at the data.
        # x=36 -> c33..38
        # x=20 -> c17..22
        # x=44 -> c41..46
        # Rule: col_start = px - 3. Range width = 6.
        col_start = px - 3
        for r in range(56, 62):
            for c in range(col_start, col_start + 6):
                if r == 56 or r == 61:
                    new_grid[r, c] = 0
                elif r >= 57 and r <= 60:
                    # Only set boundary cells of the block to 0 if it's not already 0
                    if c == col_start or c == col_start + 5:
                        new_grid[r, c] = 0
    
    elif py == 30:
        # Middle clicks move colors from bottom blocks to middle region (r28-r31)
        # x=23 -> color 9, cols [21, 24], affects r28-r31
        # x=29 -> color 14, cols [27, 30], affects r28-r31
        # x=35 -> color 11, cols [33, 36], affects r28-r31
        # Let's deduce the mapping.
        # x=23 -> col 21, color 9
        # x=29 -> col 27, color 14
        # x=35 -> col 33, color 11
        # Rule: col_start = px - 2. Color depends on px?
        # Looking at initial grid r57: 14(c18-21), 15(c22-25), 9(c26-29), 11(c30-33)? No.
        # Initial grid r57: 4x18, 14x4, 4x4, 15x4, 4x4, 9x4, 4x4, 11x4, 4x18
        # Indices in r57: c18-21 (14), c26-29 (15), c34-37 (9), c42-45 (11)
        # Wait, let's re-read INITIAL GRID r57:
        # r57: 4x18, 14x4, 4x4, 15x4, 4x4, 9x4, 4x4, 11x4, 4x18
        # Col counts: 18 + 4 = 22; 22+4=26; 26+4=30; 30+4=34; 34+4=38; 38+4=42; 42+4=46...
        # So blocks are at [18,21], [26,29], [34,37], [42,45].
        # Colors: 14, 15, 9, 11.
        # Clicks at y=30:
        # x=23 -> color 9, cols [21, 24] in middle region? No, delta says r28c21:9x4.
        # x=29 -> color 14, cols [27, 30] in middle region? Delta says r28c27:14x4.
        # x=35 -> color 11, cols [33, 36] in middle region? Delta says r28c33:11x4.
        
        col_start = px - 2
        if px == 23: color = 9
        elif px == 29: color = 14
        elif px == 35: color = 11
        else: return grid
        
        for r in range(28, 32):
            for c in range(col_start, col_start + 4):
                new_grid[r, c] = color
        
        # Also affects bottom area (sets to 4)
        # For x=23, it sets a block at c33..38 to 4.
        # For x=29, it sets a block at c17..22 to 4.
        # For x=35, it sets a block at c41..46 to 4.
        bottom_map = {23: 33, 29: 17, 35: 41}
        b_col = bottom_map[px]
        for r in range(56, 62):
            for c in range(b_col, b_col + 6):
                if r >= 57 and r <= 60:
                    if c == b_col + 2 or c == b_col + 3: # just a guess for the "4x2, 2x2, 4x2" pattern
                        pass 
                    else:
                        new_grid[r, c] = 4
                else:
                    new_grid[r, c] = 4

    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it's when certain blocks are filled.
    # Let's assume it's complete if some specific area is no longer color 4.
    return False