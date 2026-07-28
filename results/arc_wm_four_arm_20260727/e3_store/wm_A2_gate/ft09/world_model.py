import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        if grid[py, px] == 1:
            grid[py, px] = 15
            # Find the bottom-most 1 in the same column
            for r in range(grid.shape[0] - 1, -1, -1):
                if grid[r, px] == 1:
                    break
            # Move all 1s in the column down to the bottom
            for r in range(grid.shape[0] - 1, -1, -1):
                if grid[r, px] == 1:
                    grid[r, px] = 0
            for r in range(grid.shape[0] - 1, -1, -1):
                if grid[r, px] == 1:
                    grid[r, px] = 1
    return grid

def is_level_complete(grid):
    # Check if all 1s are at the bottom of their columns
    for col in range(grid.shape[1]):
        for r in range(grid.shape[0] - 1, -1, -1):
            if grid[r, col] == 1:
                if r < grid.shape[0] - 1:
                    return False
    return True