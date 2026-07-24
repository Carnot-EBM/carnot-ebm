import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game appears to be a "click to fill" or "paint" mechanic where clicking on a
    # specific region (likely the left border or a specific target area) changes the
    # color of the leftmost cell of a row to 5 (gray).
    # Observations:
    # - Action 6 is a click.
    # - Clicks at various coordinates result in changing grid[row][0] to 5.
    # - The row index seems to correspond to the y-coordinate of the click, possibly
    #   adjusted or mapped. However, looking at the data:
    #   Click y=2 -> r0c0 changes.
    #   Click y=6 -> r1c0 changes.
    #   Click y=6 -> r2c0 changes.
    #   ...
    #   This suggests a cumulative or sequential effect, or perhaps the click targets
    #   a specific "cursor" or "active row" that moves.
    #   Alternatively, it might be simpler: clicking anywhere on the left side (col 0)
    #   or a specific zone changes the cell at (row, 0) to 5.
    #   Let's look at the y-coordinates vs row indices:
    #   y=2 -> row 0
    #   y=6 -> row 1
    #   y=6 -> row 2
    #   y=6 -> row 3
    #   y=6 -> row 4
    #   y=6 -> row 5
    #   This is inconsistent for a direct mapping. It might be that the click
    #   increments a counter or moves a pointer.
    #   However, a simpler rule often found in ARC is: clicking on a cell changes it.
    #   But here, the click coordinates (23,2) changed (0,0). (59,6) changed (1,0).
    #   This implies a global state or a specific target.
    #   Given the lack of clear movement logic, and the fact that only (row, 0) changes
    #   to 5, let's assume the rule is: clicking changes the leftmost cell of the
    #   "current" row to 5. The "current" row might be tracked by a hidden state,
    #   but since we only have the grid, we must infer it.
    #   Wait, looking at the initial grid, column 0 is all 0s.
    #   The changes are always setting grid[row][0] = 5.
    #   Let's assume the click's y-coordinate determines the row, but with an offset
    #   or mapping. Or perhaps it's simpler: the click targets the first 0 in column 0
    #   from the top?
    #   Let's check:
    #   Initial: all col 0 are 0.
    #   Click 1: y=2. Changes r0. (First 0 from top is r0).
    #   Click 2: y=6. Changes r1. (First 0 from top is now r1).
    #   Click 3: y=6. Changes r2. (First 0 from top is now r2).
    #   This fits! The rule is: clicking changes the topmost cell in column 0 that is
    #   not 5 (or is 0) to 5.
    
    new_grid = grid.copy()
    if action == 6:
        # Find the first row where grid[row, 0] is not 5 (or is 0)
        for r in range(grid.shape[0]):
            if new_grid[r, 0] != 5:
                new_grid[r, 0] = 5
                break
    return new_grid

def is_level_complete(grid):
    # The level is complete when all cells in column 0 are 5?
    # Or perhaps when a specific pattern is formed.
    # Given the limited data, let's assume the goal is to fill column 0 with 5s.
    # However, usually ARC tasks have a more visual goal.
    # Let's check if there are any other changes. The deltas only show changes in col 0.
    # So the win condition might be when col 0 is all 5s.
    return np.all(grid[:, 0] == 5)