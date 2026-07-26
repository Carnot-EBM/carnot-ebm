import numpy as np

def engine(grid, action, data):
    if action == 6 and data is not None:
        px, py = data['x'], data['y']
        # Convert pixel coordinates to logical grid coordinates
        gx, gy = px // 1, py // 1
        # Check if the click is on a 0 cell (empty space)
        if grid[gy, gx] == 0:
            # Create a copy of the grid
            new_grid = grid.copy()
            # Set the clicked cell to 5
            new_grid[gy, gx] = 5
            # Set the cell directly below the clicked cell to 5
            if gy + 1 < grid.shape[0]:
                new_grid[gy + 1, gx] = 5
            # Set the cell directly to the right of the clicked cell to 5
            if gx + 1 < grid.shape[1]:
                new_grid[gy, gx + 1] = 5
            return new_grid
        else:
            # If the clicked cell is not empty, return the grid unchanged
            return grid.copy()
    else:
        # For other actions, return the grid unchanged
        return grid.copy()

def is_level_complete(grid):
    # Check if the bottom row (row 63) is completely filled with 5s
    return np.all(grid[63, :] == 5)