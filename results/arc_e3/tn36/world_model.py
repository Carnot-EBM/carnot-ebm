import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if not (0 <= py < grid.shape[0] and 0 <= px < grid.shape[1]):
            return grid
        # Convert pixel coordinates to logical coordinates
        logical_py = py // 1
        logical_px = px // 1
        # Apply the change: set the cell to color 1
        grid_copy = grid.copy()
        grid_copy[logical_py, logical_px] = 1
        return grid_copy
    return grid

def is_level_complete(grid):
    # Check if the grid is full of color 1
    return np.all(grid == 1)