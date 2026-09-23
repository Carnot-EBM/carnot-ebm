import numpy as np


def engine(grid, action, data):
    grid = grid.copy()
    if action == 6 and data is not None:
        px, py = data['x'], data['y']
        x, y = px // 1, py // 1
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            # Find the 6x6 block containing (x, y)
            bx = (x // 6) * 6
            by = (y // 6) * 6
            # Check if all cells in the 6x6 block are color 9
            block = grid[by:by+6, bx:bx+6]
            if block.size > 0 and np.all(block == 9):
                grid[by:by+6, bx:bx+6] = 8
                # Update bottom row indicator
                col_idx = bx + 3
                if 0 <= col_idx < grid.shape[1]:
                    grid[-1, col_idx] = 11
                    grid[-1, col_idx + 1] = 11
    return grid


def is_level_complete(grid):
    # Win when no color-9 blocks remain inside the main area (rows 0-62)
    return not np.any(grid[:63, :] == 9)