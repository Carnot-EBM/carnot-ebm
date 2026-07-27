import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 3:
        if data is not None:
            px, py = data['x'], data['y']
            # Toggle cell at pixel coordinates (logical * 1)
            new_grid[py, px] = 1 - grid[py, px]
    elif action == 2:
        # Gravity: move all non-7 cells down
        for col in range(W):
            col_data = grid[:, col]
            # Find non-7 cells
            non_7 = col_data[col_data != 7]
            # Find 7 cells
            seven = col_data[col_data == 7]
            # Reconstruct column
            new_col = np.zeros(H, dtype=int)
            idx = 0
            for val in seven:
                new_col[idx] = val
                idx += 1
            for val in non_7:
                new_col[idx] = val
                idx += 1
            new_grid[:, col] = new_col
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all cells are 7
    return np.all(grid == 7)