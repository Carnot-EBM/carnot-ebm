import numpy as np

def engine(grid, action, data):
    if action == 2:
        # Action 2: Move right
        new_grid = grid.copy()
        # Find all cells with color 11 (the player)
        player_positions = np.argwhere(grid == 11)
        for r, c in player_positions:
            # Check if there is space to the right
            if c < grid.shape[1] - 1 and grid[r, c + 1] == 0:
                new_grid[r, c + 1] = 11
                new_grid[r, c] = 0
        return new_grid
    elif action == 3:
        # Action 3: Move down
        new_grid = grid.copy()
        player_positions = np.argwhere(grid == 11)
        for r, c in player_positions:
            if r < grid.shape[0] - 1 and grid[r + 1, c] == 0:
                new_grid[r + 1, c] = 11
                new_grid[r, c] = 0
        return new_grid
    elif action == 4:
        # Action 4: Move left
        new_grid = grid.copy()
        player_positions = np.argwhere(grid == 1)
        for r, c in player_positions:
            if c > 0 and grid[r, c - 1] == 0:
                new_grid[r, c - 1] = 1
                new_grid[r, c] = 0
        return new_grid
    elif action == 7:
        # Action 7: Move up
        new_grid = grid.copy()
        player_positions = np.argwhere(grid == 1)
        for r, c in player_positions:
            if r > 0 and grid[r - 1, c] == 0:
                new_grid[r - 1, c] = 1
                new_grid[r, c] = 0
        return new_grid
    elif action == 6:
        # Action 6: Click
        if data is not None:
            px, py = data['x'], data['y']
            logical_x, logical_y = px // 1, py // 1
            if 0 <= logical_y < grid.shape[0] and 0 <= logical_x < grid.shape[1]:
                grid_copy = grid.copy()
                grid_copy[logical_y, logical_x] = 0
                return grid_copy
        return grid.copy()
    else:
        return grid.copy()

def is_level_complete(grid):
    # Check if all cells are filled with non-zero values
    return np.all(grid != 0)