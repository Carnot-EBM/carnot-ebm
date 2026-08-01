import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where ACTION2 moves an object or changes state in a specific pattern.
    # Based on the observed transitions, ACTION2 shifts some colors/patterns.
    # Let's analyze the patterns:
    # r8c14:5x5 r8c20:2x5 ... etc.
    # This looks like it's moving a "block" of color 2 and color 5.
    # Action 2 is likely 'right'.
    # Action 4 is likely 'left'.
    # Action 1 is 'up', Action 3 is 'down'.
    # Looking at the observed transitions for ACTION2:
    # It repeatedly fills areas with color 5 and replaces others with color 9.
    # In each step of ACTION2, the x-coordinates are shifting right by 6 units (e.g., c14->c20->c26->c32).
    # Shift amount is 6.
    # The target area is being modified based on a<|channel>thought
    # that theres a block of size 5x5.
    # However, the provided deltas are very complex.
    # Let's try to implement a simple movement logic if we can**
    # grid[r, c] = value
    # Based on the data, ACTION2 shifts things right.
    # ACTION4 shifts thing left.
    # Action 2: Right, Action 4: Left, Action 1: Up, Action 3: Down.
    # But it's not just moving an object; it' same as "painting" or "toggling".
    # la//
    #
    # Let's look at the INITIAL GRID again.
    # There is a large region of color 5 (obj6) from r7 to r56.
    # Inside this region, there are holes/objects of other colors.
    # It seems like there's a "cursor" or "brush" that moves and changes colors.
    # In each ACTION2 transition, a 5x5 block of color 5 replaces something,
    # and a 5x5 block of color 9 replaces others.
    # The cursor position starts around (8, 14).
    # After one ACTION2, it moves to (8, 20). Then (8, 26), then (8, 32).
    # This confirms a shift of 6 pixels per action.
    #
    # Let's try to implement a simple movement model for the cursor.
    # We can find the current "active" area by looking for specific patterns.
    # Since we don't have the full state history, we'll assume the cursor
    # is tracked internally or derived from the grid.
    # But the engine must be pure.
    # So we can actually identify the cursor based on the only place where color 2 exists?
    # No, color 2 is at (1,1) to (3,3).
    # Wait, in the deltas: r8c20:2x5 means row 8, col 20, value 2, count 5.
    # So color 2 IS moving!
    # Initial grid has no color 2 at (8, 14). It's there after Action 2.
    # Actually, let's look closer: r8c14:5x5 r8c20:2x5.
    # Row 8, Col 14 gets color 5 (count 5), then Col 20 gets color 2 (count 5).
    # This means a block of size 5x5 of color 2 is moving right by 6 units.
    # The "brush" consists of a 5x5 block of color 2 and some other changes.
    # Let's see what happens to the background.
    # When the brush moves, it leaves behind color 5 and replaces something with color 9.
    # Specifically, ACTION2 shifts the cursor (r, c) -> (r, c+6).
    # ACTION4 shifts the cursor (r, c) -> (r, c-6).
    # ACTION1 shifts (r, c) -> (r-6, c).
    # ACTION3 shifts (r, c) -> (r+6, c).
    #
    # Now we need to find the initial cursor position.
    # In the first transition, Action 2 starts at (8, 14) and moves to (8, 20).
    # So the cursor was at (8, 14)? No, if it moved TO (8, 20), it started at (8, 14).
    # Wait: "changed cells... r8c14:5x5 r8c20:2x5". This means at col 14 it became 5, and at col 20 it became 2.
    # This is exactly a block of size 5 moving from 14 to 20.
    # The brush is a 5x5 block of color 2.
    # When it moves, it replaces its old position with color 5.
    # Also, there's some interaction with other colors (like color 9 appearing).
    # Let's look at the deltas for ACTION2 again:
    # r14c14:9x5 ... r17c14:9x5. These are rows 14-18.
    # It seems when the brush (color 2) is at some position, it affects another area?
    # Or maybe there are multiple brushes?
    # Actually, looking at the object structure, obj8 (color 8) is also changing.
    # This looks like a game where you move a block to "clear" or "paint" an area.
    # Given the complexity and the limited data, the most likely rule is:
    # Action 2: Right, 4: Left, 1: Up, 3: Down.
    # Each action shifts a specific pattern by 6 units.
    # But since we must return a grid, let's try to find the blocks of color 2 and shift them.

    new_grid = grid.copy()
    
    # Find all blocks of color 2
    coords = np.argwhere(grid == 2)
    if coords.size == 0:
        return new_grid
    
    # Assume they move as a group
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    dy, dx = 0, 0
    if action == 1: dy = -6
    elif action == 2: dx = 6
    elif action == 3: dy = 6
    elif action == 4: dx = -6
    
    # The brush (color 2) moves, and its old position becomes color 5
    # We only apply this if it's within bounds
    for r in range(y_min, y_max + 1):
        for c in range(x_min, x_max + 1):
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                new_grid[r, c] = 5 # Old position becomes background
                nr, nc = r + dy, c + dx
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    new_grid[nr, nc] = 2 # New position is the brush
                    
    return new_grid

def is_level_complete(grid):
    # Level complete usually means some target state is reached.
    # In many ARC games, it's when a certain color is gone or a pattern is formed.
    # Looking at the INITIAL GRID, there's a line of color 9 at the bottom (r63).
    # And a small block of color 1 at (63, 62).
    # The deltas show r63c61:1x1, r63c60:1x1... this means color 1 is moving left!
    # This suggests the goal is to move the color 1 block to the far left?
    # Or maybe clear all blocks of color 8?
    # Let's check if any color 8 remains.
    return not np.any(grid == 8)