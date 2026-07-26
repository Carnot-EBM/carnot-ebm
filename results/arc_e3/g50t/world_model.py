import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move player down
        if data is not None:
            # Click action
            px, py = data['x'], data['y']
            new_grid[py, px] = 1
        else:
            # Directional move
            # Find player
            player_pos = np.argwhere(new_grid == 1)
            if len(player_pos) > 0:
                r, c = player_pos[0]
                if r < H - 1:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c] = 1
    elif action == 2:
        # Move player up
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 1
        else:
            player_pos = np.argwhere(new_grid == 1)
            if len(player_pos) > 0:
                r, c = player_pos[0]
                if r > 0:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c] = 1
    elif action == 3:
        # Move player left
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 1
        else:
            player_pos = np.argwhere(new_grid == 1)
            if len(player_pos) > 0:
                r, c = player_pos[0]
                if c > 0:
                    new_grid[r, c] = 0
                    new_grid[r, c - 1] = 1
    elif action == 4:
        # Move player right
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 1
        else:
            player_pos = np.argwhere(new_grid == 1)
            if len(player_pos) > 0:
                r, c = player_pos[0]
                if c < W - 1:
                    new_grid[r, c] = 0
                    new_grid[r, c + 1] = 1
    elif action == 5:
        # Toggle color at player position
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 1
        else:
            player_pos = np.argwhere(new_grid == 1)
            if len(player_pos) > 0:
                r, c = player_pos[0]
                new_grid[r, c] = 1
    elif action == 6:
        # Click action
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 1
    elif action == 7:
        # Toggle color at player position
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 1
        else:
            player_pos = np.argwhere(new_grid == 1)
            if len(player_pos) > 0:
                r, c = player_pos[0]
                new_grid[r, c] = 1
    
    return new_grid

def is_level_complete(grid):
    # Check if the grid is complete (all cells filled)
    return np.all(grid != 0)