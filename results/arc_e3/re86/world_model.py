import numpy as np

import numpy as np

def engine(grid, action, data):
    grid = grid.copy()
    if action == 1:
        # Action 1: Fill vertical column with 9s, then fill horizontal row with 5s, then fill diagonal with 1s
        # Based on observed transitions, action 1 seems to fill a vertical column at a specific position
        # with 9s, then a horizontal row with 5s, and finally a diagonal with 1s
        # The exact positions are determined by the grid state
        # From the transitions, it appears action 1 fills a vertical column at column 33 with 9s
        # and a horizontal row at row 45 with 5s, and a diagonal with 1s
        # We need to determine the exact positions from the grid state
        # Based on the pattern, action 1 fills a vertical column at column 33 with 9s
        # and a horizontal row at row 45 with 5s, and a diagonal with 1s
        # The diagonal seems to be from (45, 33) to (63, 63)
        # Let's implement this logic
        # Fill vertical column at column 33 with 9s
        grid[:, 33] = 9
        # Fill horizontal row at row 45 with 5s
        grid[45, :] = 5
        # Fill diagonal from (45, 33) to (63, 63) with 1s
        for i in range(19):
            grid[45 + i, 33 + i] = 1
    elif action == 2:
        # Action 2: Similar to action 1 but with different values
        # Fill vertical column at column 33 with 5s
        grid[:, 33] = 5
        # Fill horizontal row at row 45 with 5s
        grid[45, :] = 5
        # Fill diagonal from (45, 33) to (63, 63) with 1s
        for i in range(19):
            grid[45 + i, 33 + i] = 1
    elif action == 3:
        # Action 3: Fill vertical column at column 33 with 9s, then fill horizontal row at row 45 with 5s
        # Fill vertical column at column 33 with 9s
        grid[:, 33] = 9
        # Fill horizontal row at row 45 with 5s
        grid[45, :] = 5
    elif action == 4:
        # Action 4: Fill a vertical column at column 21 with 5s, then fill a horizontal row at row 27 with 5s
        # Fill vertical column at column 21 with 5s
        grid[:, 21] = 5
        # Fill horizontal row at row 27 with 5s
        grid[27, :] = 5
        # Fill a diagonal from (27, 21) to (38, 32) with 11s
        for i in range(12):
            grid[27 + i, 21 + i] = 11
    elif action == 5:
        # Action 5: Fill a horizontal row at row 27 with 0s, then fill a vertical column at column 33 with 9s
        # Fill horizontal row at row 27 with 0s
        grid[27, :] = 0
        # Fill vertical column at column 33 with 9s
        grid[:, 33] = 9
    elif action == 6:
        # Action 6: Click at pixel coordinates
        # Convert pixel coordinates to logical coordinates
        px = data['x']
        py = data['y']
        row = py // 1
        col = px // 1
        # Set the cell at (row, col) to 1
        grid[row, col] = 1
    elif action == 7:
        # Action 7: Not observed in the transitions, but we'll implement a default behavior
        # For now, we'll assume it does nothing
        pass
    return grid

def is_level_complete(grid):
    # Check if the grid is a win state
    # Based on the observed transitions, a win state is when the grid has a specific pattern
    # From the transitions, it appears that a win state is when the grid has a diagonal of 1s from (45, 33) to (63, 63)
    # and a horizontal row of 5s at row 45, and a vertical column of 9s at column 33
    # We'll check for this pattern
    # Check if the diagonal from (45, 33) to (63, 63) has 1s
    for i in range(19):
        if grid[45 + i, 33 + i] != 1:
            return False
    # Check if the horizontal row at row 45 has 5s
    if not np.all(grid[45, :] == 5):
        return False
    # Check if the vertical column at column 33 has 9s
    if not np.all(grid[:, 33] == 9):
        return False
    return True

def is_level_complete(grid):
    import numpy as np
    g = np.array(grid)
    if g.size == 0:
        return False
    return np.all(g == 0)
