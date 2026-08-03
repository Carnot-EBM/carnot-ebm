import numpy as np

def engine(grid, action, data):
    """
    Induces the world model for ar25 based on observed transitions.
    The grid contains several regions and a marker (color 11) at column 63.
    ACTION2 seems to move a pattern downwards and shift the marker in col 63.
    ACTION3 seems to shift patterns horizontally.
    """
    new_grid = grid.copy()
    h, w = new_grid.shape

    if action == 2:
        # ACTION2 shifts specific blocks of colors (5 and 4) down by 3 rows
        # and moves the color 5 marker in column 63 down by 1 row.
        
        # Marker movement in col 63
        for r in range(h - 1, -1, -1):
            if new_grid[r, 63] == 5:
                new_grid[r, 63] = 11 # Restore original background if it was 5
                if r + 1 < h:
                    new_grid[r+1, 63] = 5
                break
        
        # Pattern shifting logic:
        # Based on deltas, there are two vertical columns of "active" areas.
        # Left side around col 3-12, Right side around col 51-60.
        # The observed delta shows a block moving from [15-17, 3-11] to [18-20, 3-11], etc.
        # This is a shift of 3 rows.
        
        # We identify regions of color 5 or 4 that aren't part of the static layout (color 10)
        # and move them down by 3 units.
        temp_grid = grid.copy()
        for r in range(h - 3):
            for c in range(w):
                val = temp_grid[r, c]
                if val in [4, 5]:
                    # Check if this cell is being replaced by something else in the delta
                    # In ar25, it looks like blocks of size 3x9 are sliding.
                    new_grid[r + 3, c] = val
                    # If we moved it, clear the old one unless it's background/static
                    # But based on deltas, they overwrite with color 9 (background).
                    if val != 10:
                        new_grid[r, c] = 9

    elif action == 3:
        # ACTION3 shifts patterns horizontally.
        # Observed: r15c6 -> r15c3 (shift left 3) then r15c3 -> r15c?
        # It seems to shift the active block left or right.
        # Let's implement a simple horizontal shift for colors 4 and 5.
        shift = -3 # Default observed shift was left
        temp_grid = grid.copy()
        for r in range(h):
            for c in range(w):
                val = temp_grid[r, c]
                if val in [4, 5]:
                    nc = c + shift
                    if 0 <= nc < w:
                        new_grid[r, nc] = val
                        if val != 10:
                            new_grid[r, c] = 9
                    else:
                        # Wrap around or clamp? Usually ARC is clamped or specific.
                        pass

    return new_grid

def is_level_complete(grid):
    """
    The win state isn't explicitly provided, but typically it involves 
    reaching a target position or clearing a pattern.
    Given the marker at col 63 moves down, maybe reaching the bottom is the goal.
    """
    # Check if color 5 reached the bottom of column 63
    if grid[-1, 63] == 5:
        return True
    return False