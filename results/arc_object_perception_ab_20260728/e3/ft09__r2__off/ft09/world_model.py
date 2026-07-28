import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        # Apply the observed rule: change a 6x6 block to color 8
        # The block is centered at (px, py)
        for r in range(py - 3, py + 4):
            for c in range(px - 3, px + 4):
                if 0 <= r < H and 0 <= c < W:
                    grid[r, c] = 8
        return grid
    return grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # The win state has specific patterns in the first 32 rows and last row
    # Based on the observed win state, we check for the presence of color 4 in the first 32 rows
    # and color 12 in the last row
    for r in range(grid.shape[0]):
        if r < 32:
            if not np.any(grid[r, :] == 4):
                return False
        elif r == 63:
            if not np.any(grid[r, :] == 12):
                return False
    return True