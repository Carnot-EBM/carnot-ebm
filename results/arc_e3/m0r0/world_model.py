import numpy as np

def engine(grid, action, data):
    if action == 1:
        # Move right
        return move(grid, 1)
    elif action == 2:
        # Move left
        return move(grid, -1)
    elif action == 3:
        # Move down
        return move(grid, 1, axis=1)
    elif action == 4:
        # Move up
        return move(grid, -1, axis=1)
    elif action == 5:
        # Toggle
        return toggle(grid)
    elif action == 6:
        # Click (pixel)
        if data:
            px, py = data['x'], data['y']
            logical_x, logical_y = px // 1, py // 1
            if 0 <= logical_y < grid.shape[0] and 0 <= logical_x < grid.shape[1]:
                grid = grid.copy()
                grid[logical_y, logical_x] = 10
                return grid
        return grid
    elif action == 7:
        # Reset
        return grid.copy()
    return grid

def move(grid, direction, axis=0):
    grid = grid.copy()
    if axis == 0:
        # Horizontal movement
        if direction == 1:
            # Move right
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1] - 1, 0, -1):
                    if grid[r, c] != 0 and grid[r, c] != 11:
                        grid[r, c] = grid[r, c - 1]
                        grid[r, c - 1] = 0
        else:
            # Move left
            for r in range(grid.shape[0]):
                for c in range(1, grid.shape[1]):
                    if grid[r, c] != 0 and grid[r, c] != 11:
                        grid[r, c] = grid[r, c + 1]
                        grid[r, c + 1] = 0
    else:
        # Vertical movement
        if direction == 1:
            # Move down
            for c in range(grid.shape[1]):
                for r in range(grid.shape[0] - 1, 0, -1):
                    if grid[r, c] != 0 and grid[r, c] != 11:
                        grid[r, c] = grid[r + 1, c]
                        grid[r + 1, c] = 0
        else:
            # Move up
            for c in range(grid.shape[1]):
                for r in range(1, grid.shape[0]):
                    if grid[r, c] != 0 and grid[r, c] != 11:
                        grid[r, c] = grid[r - 1, c]
                        grid[r - 1, c] = 0
    return grid

def toggle(grid):
    grid = grid.copy()
    # Toggle all 5s to 0s and 0s to 5s
    grid[grid == 5] = 0
    grid[grid == 0] = 5
    return grid

def is_level_complete(grid):
    # Check if all 5s are converted to 0s
    return np.all(grid[grid == 5] == 0)