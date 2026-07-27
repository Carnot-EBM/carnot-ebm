import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        if grid[py, px] == 9:
            grid[py, px] = 15
            # Apply gravity to the column
            col = grid[:, px]
            non_empty = col[col != 9]
            empty_count = np.sum(col == 9)
            if empty_count > 0:
                grid[py:py+empty_count, px] = 15
                grid[py+empty_count:, px] = 9
    return grid

def is_level_complete(grid):
    return False