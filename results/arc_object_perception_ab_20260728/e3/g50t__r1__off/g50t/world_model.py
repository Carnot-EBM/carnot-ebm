import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 2:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        # Determine direction
        if py < 32:
            direction = -1
        else:
            direction = 1
        # Determine target row
        if direction == -1:
            target_row = py - 32
        else:
            target_row = py - 32
        # Apply gravity
        for r in range(H):
            if r == target_row:
                continue
            row = grid[r].copy()
            # Count non-zero elements
            non_zero = np.count_nonzero(row)
            # Count zero elements
            zero_count = np.count_nonzero(row == 0)
            # Create new row with zeros at the end
            new_row = np.zeros(W, dtype=int)
            new_row[:non_zero] = row[:non_zero]
            grid[r] = new_row
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if the grid matches the win state pattern
    # The win state has a specific pattern of colors
    # Check if the grid has the correct number of non-zero elements
    non_zero_count = np.count_nonzero(grid)
    if non_zero_count != 396:
        return False
    # Check if the grid has the correct pattern
    # The win state has a specific pattern of colors
    # Check if the grid has the correct number of rows with non-zero elements
    rows_with_non_zero = np.count_nonzero(np.any(grid != 0, axis=1))
    if rows_with_non_zero != 56:
        return False
    # Check if the grid has the correct number of columns with non-zero elements
    cols_with_non_zero = np.count_nonzero(np.any(grid != 0, axis=0))
    if cols_with_non_zero != 56:
        return False
    return True