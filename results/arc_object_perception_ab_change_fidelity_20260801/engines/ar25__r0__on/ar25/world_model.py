import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid is HxW int array
    # Action 2: Move Down
    # Action 3: Move Left/Right? No, looking at ACTION3 transitions, it's a shift in x coordinates.
    # Let's analyze the ACTION3 shifts:
    # r15c6:5x3 (was r15c9:5x9) -> shifted left by 3.
    # r15c15:9x3 (was r15c12:9x12) -> shifted right by 3.
    # Looking at the observed transitions, ACTION3 seems to be "Move Right" or "Move Left".
    # Actually, let's look at the ACTION3 delta again:
    # First ACTION3: r15c6:5x3... r18c12:5x3...
    # Second ACTION3: r15c3:5x3... r18c9:5x3...
    # It looks like they are moving an object of color 5 and others.
    # The objects being moved are those contained within the larger background blocks of color 9.
    # In this case, there is a block of color 5 on the left side and a block of color 4/11 on the right side.
    # Action 2 moves them down.
    # Action 3 moves them horizontally.
    
    # Let's refine the action mapping based on common ARC patterns:
    # Action 2: Down
    # Action 3: Right? No, looking at the deltas, it's shifting coordinates.
    # Wait, the ACTION3 shifts are symmetric.
    # If we move the 'player' (color 5) and its mirror image (color 4), they shift in opposite directions or same direction.
    # Let's check:
    # Initial obj5 (color 5): bbox=(15, 9, 23, 17). After first ACTION3: shifted to c6. Shift = -3.
    # Initial obj6 (color 4): bbox=(15, 45, 23, 53). After first ACTION3: shifted to c45... wait, no change?
    # Actually, let's look at r15c45:9x3. The original was r15c45:4x9. Now it's 9x3. This means color 4 is being replaced by color 9.
    # It looks like a "sliding puzzle" where objects of color 5 and 4/11 slide within their respective regions.
    
    # grid_new = grid.copy()
    # But based on the observed transitions, the most critical part is the laout:
    # There is a central divider (color 10).
    # Left side (col 0-29) and Right side (col 33-62).
    # Action 2 moves things down.
    # Action 3 moves things horizontally.
    
    # Let's try to implement a a simple movement model for the 'active' objects.
    # active_colors = [5, 4, 11]
    # We actually see that as the object of color 5 moves, its mirror image of color 4 or 11 also moves.
    # In ACTION2, they move down together.
    # In ACTION3, they move left/right.
    
    # # Find the center of mass of the active objects.
    # # Find the blocks of color 5.
    # #<|channel>thought
    # Actually, let's look at the deltas again.
    # The movements are very precise shifts.
    # Action 2: Down shift by 3 rows?
    # First ACTION2: r15c3:9x9... r18c3:5x6...
    # Original obj5 was r15..r23. New is r18..r26. Shift = +3.
    # Yes, Action 2 is "Move Down" by 3 units.
    # Action 3 is "Move Left" (for the left side) and "Move Right" (for the right side)?
    # Let's check first ACTION3: obj5 shifted from c9 to c6. Shift = -3.
    # Mirror object (color 4) shifted from c45 to c48? No, r15c45:9x3 means it was replaced by background.
    # Looking at the delta: r15c45:9x3, r15c54:4x3. Original was r15c45:4x9.
    # So color 4 moved from [45, 53] to [54, 62]? Wait, no.
    # Let's just implement a simple shift for any non-background colors in the two regions.

    grid_new = grid.copy()
    
    if action == 2: # Move Down
        # Shift everything that isn't color 9 or 10 down by 3
        # We need to handle boundaries and avoid overwriting.
        # For simplicity, let's identify the 'active' objects.
        mask = (grid != 9) & (grid != 10)
        # To move down by 3:
        # New positions are old positions + (3, 0)
        # But we must clear old positions first.
        # This is tricky with numpy.
        
        # Let's use a simpler approach: find all cells of active colors and move them.
        coords = np.argwhere(mask)
        for r, c in coords:
            grid_new[r, c] = 9 if (c < 30 or c > 32) else 10
        for r, c in coords:
            nr, nc = r + 3, c
            if 0 <= nr < grid_new.shape[0]:
                grid_new[nr, nc] = grid[r, c]

    elif action == 3: # Move Left/Right
        mask = (grid != 9) & (grid != 10)
        coords = np.argwhere(mask)
        for r, c in coords:
            grid_new[r, c] = 9 if (c < 30 or c > 32) else 10
        for r, c in coords:
            # If on the left side, move left by 3. If on right side, move right by 3.
            nc = c - 3 if c < 30 else c + 3
            nr = r
            if 0 <= nc < grid_new.shape[1]:
                grid_new[nr, nc] = grid[r, c]
                
    return grid_new

def is_level_complete(grid):
    # The win state isn't explicitly given, but usually it's when objects reach a target.
    # In this game, maybe when they hit the bottom?
    # Let's check for any active colors at the very bottom row.
    active_colors = [4, 5, 11]
    bottom_row = grid[-1, :]
    return any(np.isin(bottom_row, active_colors))

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition for ar25 is that the grid contains only 
    one color (excluding background 0).
    """
    grid = np.array(grid)
    mask = grid != 0
    if not np.any(mask):
        return False
    
    unique_colors = np.unique(grid[mask])
    return len(unique_colors) == 1
