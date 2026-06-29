import numpy as np

def engine(grid, action, data):
    grid = grid.copy()
    if action == 1:
        # Action 1: Move right
        # Find all non-5 cells and move them right by 1
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1] - 1):
                if grid[r, c] != 5:
                    grid[r, c] = 5
                    grid[r, c + 1] = grid[r, c]
    elif action == 2:
        # Action 2: Move left
        # Find all non-5 cells and move them left by 1
        for r in range(grid.shape[0]):
            for c in range(1, grid.shape[1]):
                if grid[r, c] != 5:
                    grid[r, c] = 5
                    grid[r, c - 1] = grid[r, c]
    elif action == 3:
        # Action 3: Move down
        # Find all non-5 cells and move them down by 1
        for r in range(grid.shape[0] - 1):
            for c in range(grid.shape[1]):
                if grid[r, c] != 5:
                    grid[r, c] = 5
                    grid[r + 1, c] = grid[r, c]
    elif action == 4:
        # Action 4: Move up
        # Find all non-5 cells and move them up by 1
        for r in range(1, grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != 5:
                    grid[r, c] = 5
                    grid[r - 1, c] = grid[r, c]
    elif action == 5:
        # Action 5: Toggle walls (5 <-> 0)
        grid[grid == 5] = 0
        grid[grid == 0] = 5
    elif action == 6:
        # Action 6: Click (no effect on grid)
        pass
    elif action == 7:
        # Action 7: Clear all non-5 cells
        grid[grid != 5] = 5
    return grid

def is_level_complete(grid):
    # Check if there are any non-5 cells
    return np.all(grid == 5)