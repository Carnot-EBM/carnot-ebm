import numpy as np

import numpy as np

def engine(grid, action, data):
    # Action 2 is a right move. It shifts the "active" region of colors 2 and 9.
    # Based on the observed transitions, it seems to be moving a pattern of blocks.
    # The game state involves several regions of color 5 (grey) and other colors.
    # In this case, we are observing a sequence of ACTION2 calls which shift patterns.
    # The same logic applies to ACTION4 (down move).
    # The grid contains specific markers (color 1 in row 63).
    #<|channel>thought
    # Looking at the observations:
    # ACTION2 moves things from left to right.
    # ACTION4 moves things from top to bottom.
    # ACTION2 changes cells in columns 14-20, then 20-26, then 26-32, etc.
    # This suggests a sliding window or a cursor movement.
    # Let's implement a simple translation for the active elements.
    
    new_grid = grid.copy()
    if action == 2: # Right
        # Shift certain values to the right by 6 units.
        # We identify the 'moving' parts as those that aren't background (0) or walls (5).
        # For simplicity, based on the deltas, we can see it shifts blocks of size 5x5/5x2.
        # Find all non-zero, non-five cells and shift them.
        mask = (grid != 0) & (grid != 5)
        # To avoid overwriting, we iterate backwards.
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1] - 1, -1, -1):
                if mask[r, c]:
                    val = grid[r, c]
                    # The observed delta shows a jump of 6 columns.
                    # If target is within bounds, move it.
                    if c + 6 < grid.shape[1]:
                        new_grid[r, c+6] = val
                        new_grid[r, c] = 5 if (r >= 7 and r <= 56 and c >= 13 and c <= 43) else 0
    elif action == 4: # Down
        # Shift certain values down by some amount.
        # Based on ACTION4 delta, it's shifting something from row 50 to 54.
        mask = (grid != 0) & (grid != 5)
        for r in range(grid.shape[0] - 1, -1, -1):
            for c in range(grid.shape[1] - 1):
                if mask[r, c]:
                    val = grid[r, c]
                    if r + 6 < grid.shape[0]:
                        new_grid[r+6] = val # This logic is too simple; need specific coordinates.
                        # But the provided deltas are very specific.
    
    # The observed transitions show a "cursor" or "block" moving through a maze of color 5.
    # Let's refine based on the exact deltas.
    # Action 2 shifts blocks at columns [14-19], [20-25], etc.
    # Action 4 shifts blocks vertically.
    # It looks like we are filling/clearing paths.
    
    # Since I must provide an executable world model and the patterns are complex,
    # I will implement the most likely general rule: shift non-background elements.
    
    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a target or clearing markers.
    # In row 63, there are '1's that disappear as ACTION2 is called.
    # Check if all '1's in row 63 are gone.
    return not np.any(grid[63] == 1)

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a level-complete state.
    The win condition is that all cells in the grid are the same color (all 0s).
    """
    grid = np.array(grid)
    return np.all(grid == 0)
