import numpy as np

def engine(grid, action, data):
    h, w = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move player down
        if data is None:
            # Find player (color 3)
            player_pos = np.argwhere(new_grid == 3)
            if len(player_pos) > 0:
                player_row, player_col = player_pos[0]
                if player_row < h - 1:
                    new_grid[player_row + 1, player_col] = 3
                    new_grid[player_row, player_col] = 0
    elif action == 2:
        # Move player up
        if data is None:
            player_pos = np.argwhere(new_grid == 3)
            if len(player_pos) > 0:
                player_row, player_col = player_pos[0]
                if player_row > 0:
                    new_grid[player_row - 1, player_col] = 3
                    new_grid[player_row, player_col] = 0
    elif action == 3:
        # Move player left
        if data is None:
            player_pos = np.argwhere(new_grid == 3)
            if len(player_pos) > 0:
                player_row, player_col = player_pos[0]
                if player_col > 0:
                    new_grid[player_row, player_col - 1] = 3
                    new_grid[player_row, player_col] = 0
    elif action == 4:
        # Move player right
        if data is None:
            player_pos = np.argwhere(new_grid == 3)
            if len(player_pos) > 0:
                player_row, player_col = player_pos[0]
                if player_col < w - 1:
                    new_grid[player_row, player_col + 1] = 3
                    new_grid[player_row, player_col] = 0
    elif action == 5:
        # Toggle color 14 to 0
        if data is None:
            new_grid[new_grid == 14] = 0
    elif action == 6:
        # Click action
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 0
    elif action == 7:
        # Toggle color 4 to 0
        if data is None:
            new_grid[new_grid == 4] = 0
    
    return new_grid

def is_level_complete(grid):
    # Check if all 14s are removed
    return not np.any(grid == 14)