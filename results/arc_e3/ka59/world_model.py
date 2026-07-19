import numpy as np

def engine(grid, action, data):
    if action == 3:
        return apply_action_3(grid)
    elif action == 6:
        return apply_action_6(grid, data)
    else:
        return grid.copy()

def apply_action_3(grid):
    new_grid = grid.copy()
    # Action 3 toggles specific columns based on row patterns
    # Based on observed transitions, it affects columns 15, 24, etc.
    # Pattern: affects rows 30-32 and row 63
    # Columns affected seem to be every 9th column starting from 15
    cols = [15, 24, 33, 42, 51, 60]
    for col in cols:
        if col < new_grid.shape[1]:
            # Toggle specific rows in this column
            # Based on transitions: rows 30-32 and 63
            for row in [30, 31, 32, 63]:
                if row < new_grid.shape[0]:
                    new_grid[row, col] = 1 - new_grid[row, col]
    return new_grid

def apply_action_6(grid, data):
    if data is None:
        return grid.copy()
    px, py = data['x'], data['y']
    new_grid = grid.copy()
    # Action 6 is a click at pixel coordinates
    # Based on transitions, it toggles cells at specific positions
    # The effect seems to be toggling the cell at (py, px) and possibly adjacent cells
    # From observations: toggles (63, 63), (63, 62), (63, 61), (63, 60), (31, 16), (31, 28)
    # Pattern: toggles cells in row 63 and row 31 at specific columns
    # Columns affected: 63, 62, 61, 60, 16, 28
    # These seem to be related to the click position
    # Simple rule: toggle the clicked cell and cells in the same row/column
    # Based on the data, it seems to toggle specific cells
    # Let's implement based on the observed pattern
    # Toggle cell at (py, px)
    if py < new_grid.shape[0] and px < new_grid.shape[1]:
        new_grid[py, px] = 1 - new_grid[py, px]
    # Also toggle cells in the same row and column based on the pattern
    # This is a simplified version based on the observed transitions
    # Toggle cells in row 63 and row 31 at specific columns
    # This is a heuristic based on the limited observations
    # Toggle cells in the same row as the click
    if py < new_grid.shape[0]:
        for col in range(new_grid.shape[1]):
            if col != px:
                new_grid[py, col] = 1 - new_grid[py, col]
    # Toggle cells in the same column as the click
    if px < new_grid.shape[1]:
        for row in range(new_grid.shape[0]):
            if row != py:
                new_grid[row, px] = 1 - new_grid[row, px]
    return new_grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on the initial grid, rows 0-20 are all 2s
    # Rows 21-39 have a pattern
    # Rows 42-62 are all 2s
    # Row 63 is all 4s
    # A level is complete if:
    # 1. Rows 0-20 are all 2s
    # 2. Rows 21-39 have the correct pattern
    # 3. Rows 42-62 are all 2s
    # 4. Row 63 is all 4s
    # This is a simplified check based on the initial grid
    # Check if row 63 is all 4s
    if not np.all(grid[63, :] == 4):
        return False
    # Check if rows 0-20 are all 2s
    for row in range(21):
        if not np.all(grid[row, :] == 2):
            return False
    # Check if rows 42-62 are all 2s
    for row in range(42, 63):
        if not np.all(grid[row, :] == 2):
            return False
    # Check if rows 21-39 have the correct pattern
    # This is a simplified check
    # The pattern seems to be alternating 2s and 1s in certain columns
    # This is a heuristic based on the initial grid
    return True