import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 3:
        # Action 3: Toggle a vertical line of cells in a specific column
        # The column is determined by the data parameter
        # In this case, the data is a dictionary with 'x' and 'y' keys
        # The 'x' key represents the column index
        # The 'y' key represents the row index
        # The action toggles the cells in the column from the row index to the bottom
        # The toggled cells are 2 and 3
        col = data['x']
        row = data['y']
        for i in range(row, H):
            if grid[i, col] == 2:
                grid[i, col] = 3
            elif grid[i, col] == 3:
                grid[i, col] = 2
    elif action == 2:
        # Action 2: Move a block of cells in a specific direction
        # The direction is determined by the data parameter
        # In this case, the data is a dictionary with 'x' and 'y' keys
        # The 'x' key represents the column index
        # The 'y' key represents the row index
        # The action moves the cells in the column from the row index to the bottom
        # The moved cells are 2 and 3
        col = data['x']
        row = data['y']
        for i in range(row, H):
            if grid[i, col] == 2:
                grid[i, col] = 3
            elif grid[i, col] == 3:
                grid[i, col] = 2
    return grid

def is_level_complete(grid):
    # Check if the grid is complete
    # The grid is complete if all cells are 2 or 3
    return np.all((grid == 2) | (grid == 3))