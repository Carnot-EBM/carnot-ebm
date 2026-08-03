import numpy as np

def engine(grid, action, data):
    # The game seems to involve moving blocks of colors (specifically 2s and 9s) 
    # and filling areas of color 5.
    # ACTION2 moves things right/down?
    # ACTION4 moves things left/up?
    # ACTION6 is click.
    
    # Based on the same transition patterns, let's induce a simple movement logic.
    # Action 2 shifts some pattern of pixels to the right.
    # if action == 2:
    #     # Find all regions of color 2 or 9 and shift them by 5 units.
    #     # Shift everything except for color 0 (background).
    #     # Shift the same time they move in direction 2.
    #     #<|channel>thought
    #     # Looking at the deltas, r8c14:5x5 means row 8, col 14 becomes 5.
    #     # In initial grid, r8c13 was 0, r8c14 was 5. Wait, no.
    #     # Initial grid r8: 0x13, 5x1, 9x5, 5x25... so c13=5, c14-18=9, c19=5...
    #     # Delta r8c14:5x5 means cells [14,15,16,17,18] become 5.
    #     # This effectively "erases" the block of 9s at c14-18 and replaces it with 5s.
    #     # Then r8c20:2x5 means cells [20,21,22,23,24] become 2s.
    #     # So a block of size (5,5) or similar moved from (8,14) to (8,20).
    #     # The distance is 6 columns.
    
    # Let's refine this: ACTION2 shifts blocks of color 2/9 by 6 units right.
    # ACTION4 shifts them left? Or maybe Action 2 is 'right', Action 4 is 'left'.
    # Looking at ACTION4 delta: r50c14:5x5 (becomes 5), r50c20:9x5 (becomes 9).
    # This looks like a block of 9s shifted from col 20 back to col 14.
    # Distance = 6.
    
    # We need to identify which blocks are moving.
    # In initial grid, there are blocks of 9s at (1,3), (2,5), etc.
    # But the deltas focus on rows 8-54 and cols 14-40.
    # laout: c13=5, then some pattern.
    
    # Simple implementation: shift all non-zero pixels in specific regions by 6.
    new_grid = grid.copy()
    if action == 2:
        # Shift Right by 6
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1] - 6):
                if grid[r, c] != 0 and grid[r, c] != 5:
                    # Move value to new position, old position becomes 5 or 0
                    val = grid[r, c]
                    new_grid[r, c + 6] = val
                    # If it was part of a block that is now "empty", fill with 5 if possible
                    if grid[r, c+6] == 5: # only overwrite 5s?
                        pass
                    # The observed delta shows cells becoming 5.
                    new_grid[r, c] = 5 if (r >= 7 and r <= 56) else 0
        
        # Special case for the counter at r63c60 etc.
        # ACTION2 reduces the count of 9s at the bottom right?
        # r63 has 9x62, 1x2. Delta r63c61:1x1 means cell 61 becomes 1.
        # This looks like a progress bar moving left.
        if new_grid[63, 62] == 1:
            new_grid[63, 61] = 1
            new_grid[63, 62] = 9
        elif new_grid[63, 61] == 1:
             new_grid[63, 60] = 1
             new_grid[63, 61] = 9

    elif action == 4:
        # Shift Left by 6
        for r in range(6, grid.shape[0]):
            for c in range(6, grid.shape[1]):
                if grid[r, c] != 0 and grid[r, c] != 5:
                    val = grid[r, c]
                    new_grid[r, c - 6] = val
                    new_grid[r, c] = 5 if (r >= 7 and r <= 56) else 0
                    
    return new_grid

def is_level_complete(grid):
    # Level complete when the "progress bar" at row 63 reaches a certain point?
    # Or all blocks are moved to a target area.
    # In observed transitions, it's not yet complete.
    # Usually, completion is when some specific color disappears or moves.
    return np.sum(grid == 1) > 0 and grid[63, 0] == 1