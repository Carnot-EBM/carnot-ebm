import numpy as np

def engine(grid, action, data):
    if action == 1:
        # Action 1: Click at pixel coordinates (data['x'], data['y'])
        # Convert pixel to logical coordinates
        px = data['x']
        py = data['y']
        row = py // 1
        col = px // 1
        # Set the cell to 0
        grid[row, col] = 0
        return grid
    elif action == 2:
        # Action 2: Move Up
        # Shift all non-5 cells up by 1 row
        # 5 is the background color
        new_grid = grid.copy()
        for c in range(grid.shape[1]):
            col = new_grid[:, c]
            # Find non-5 cells
            non_bg = col[col != 5]
            # Shift up
            new_col = np.zeros_like(col)
            new_col[-len(non_bg):] = non_bg
            new_grid[:, c] = new_col
        return new_grid
    elif action == 3:
        # Action 3: Move Down
        # Shift all non-5 cells down by 1 row
        new_grid = grid.copy()
        for c in range(grid.shape[1]):
            col = new_grid[:, c]
            non_bg = col[col != 5]
            new_col = np.zeros_like(col)
            new_col[:len(non_bg)] = non_bg
            new_grid[:, c] = new_col
        return new_grid
    elif action == 4:
        # Action 4: Move Left
        # Shift all non-5 cells left by 1 column
        new_grid = grid.copy()
        for r in range(grid.shape[0]):
            row = new_grid[r, :]
            non_bg = row[row != 5]
            new_row = np.zeros_like(row)
            new_row[-len(non_bg):] = non_bg
            new_grid[r, :] = new_row
        return new_grid
    elif action == 5:
        # Action 5: Move Right
        # Shift all non-5 cells right by 1 column
        new_grid = grid.copy()
        for r in range(grid.shape[0]):
            row = new_grid[r, :]
            non_bg = row[row != 5]
            new_row = np.zeros_like(row)
            new_row[:len(non_bg)] = non_bg
            new_grid[r, :] = new_row
        return new_grid
    elif action == 6:
        # Action 6: Click at pixel coordinates
        px = data['x']
        py = data['y']
        row = py // 1
        col = px // 1
        # Toggle the cell value
        grid[row, col] = 1 - grid[row, col]
        return grid
    elif action == 7:
        # Action 7: Rotate 90 degrees clockwise
        new_grid = np.rot90(grid, k=-1, axes=(0, 1))
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on the observed transitions, the win state is when all non-5 cells are 0
    # and the grid is in a specific configuration
    # For simplicity, we check if all non-5 cells are 0
    non_bg = grid[grid != 5]
    return len(non_bg) == 0