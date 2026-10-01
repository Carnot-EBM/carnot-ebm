import numpy as np

def engine(grid, action, data):
    new_grid = grid.copy()
    if action == 6 and data is not None:
        px, py = data['x'], data['y']
        # Check if click is on column 0 (the left wall)
        if px == 0:
            for r in range(64):
                if new_grid[r, 0] != 5:
                    new_grid[r, 0] = 5
    return new_grid

def is_level_complete(grid):
    # Win condition: all cells in column 0 are color 5
    return bool(np.all(grid[:, 0] == 5))