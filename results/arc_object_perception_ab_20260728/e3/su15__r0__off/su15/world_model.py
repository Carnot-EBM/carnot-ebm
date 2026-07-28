import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        h, w = grid.shape
        # Identify the 4x4 block at (py-3, px)
        block = grid[py-3:py+1, px:px+4]
        # Check if the block is uniform (all same value)
        if np.all(block == block[0, 0]):
            # Set the block to 0
            grid[py-3:py+1, px:px+4] = 0
            # Set the target cell (py, px+5) to 5
            grid[py, px+5] = 5
        return grid
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    # Check if the bottom row is all 0
    if not np.all(grid[-1, :] == 0):
        return False
    # Check if the top row is all 5
    if not np.all(grid[0, :] == 5):
        return False
    # Check if the leftmost column (excluding top and bottom) is all 5
    if not np.all(grid[1:-1, 0] == 5):
        return False
    # Check if the rightmost column (excluding top and bottom) is all 5
    if not np.all(grid[1:-1, -1] == 5):
        return False
    # Check if the second row is all 5
    if not np.all(grid[1, :] == 5):
        return False
    # Check if the second to last row is all 5
    if not np.all(grid[-2, :] == 5):
        return False
    return True