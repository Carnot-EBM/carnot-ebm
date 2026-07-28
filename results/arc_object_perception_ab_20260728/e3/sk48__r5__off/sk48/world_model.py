import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move player (color 5) down by 1
        # Find player position
        player_pos = np.argwhere(grid == 5)
        if len(player_pos) > 0:
            # Move player down
            player_pos = player_pos + 1
            # Check for collision
            if np.any((player_pos[:, 0] < H) & (player_pos[:, 1] < W) & (grid[player_pos[:, 0], player_pos[:, 1]] != 0)):
                # Collision detected, move player up
                player_pos = player_pos - 1
            # Update grid
            new_grid[player_pos[:, 0], player_pos[:, 1]] = 5
            # Clear old position
            old_pos = player_pos - 1
            new_grid[old_pos[:, 0], old_pos[:, 1]] = 0
            
    elif action == 3:
        # Action 3: Move player (color 5) right by 1
        player_pos = np.argwhere(grid == 5)
        if len(player_pos) > 0:
            player_pos = player_pos + np.array([0, 1])
            if np.any((player_pos[:, 0] < H) & (player_pos[:, 1] < W) & (grid[player_pos[:, 0], player_pos[:, 1]] != 0)):
                player_pos = player_pos - np.array([0, 1])
            new_grid[player_pos[:, 0], player_pos[:, 1]] = 5
            old_pos = player_pos - np.array([0, 1])
            new_grid[old_pos[:, 0], old_pos[:, 1]] = 0
            
    elif action == 4:
        # Action 4: Move player (color 5) left by 1
        player_pos = np.argwhere(grid == 5)
        if len(player_pos) > 0:
            player_pos = player_pos - np.array([0, 1])
            if np.any((player_pos[:, 0] < H) & (player_pos[:, 1] < W) & (grid[player_pos[:, 0], player_pos[:, 1]] != 0)):
                player_pos = player_pos + np.array([0, 1])
            new_grid[player_pos[:, 0], player_pos[:, 1]] = 5
            old_pos = player_pos + np.array([0, 1])
            new_grid[old_pos[:, 0], old_pos[:, 1]] = 0
            
    elif action == 6:
        # Action 6: Click action
        px, py = data['x'], data['y']
        # Toggle cell color
        if new_grid[px, py] == 5:
            new_grid[px, py] = 0
        else:
            new_grid[px, py] = 5
            
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all cells are either 5 or 4
    if np.any((grid != 5) & (grid != 4)):
        return False
    # Check if there are no 0s
    if np.any(grid == 0):
        return False
    return True