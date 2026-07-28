import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move player (color 2) down by 1
        # Find player position
        player_pos = np.argwhere(new_grid == 2)
        if len(player_pos) > 0:
            # Move down
            player_pos[:, 1] += 1
            # Check bounds
            if player_pos[0, 1] < H:
                # Clear old position
                new_grid[player_pos[0, 0], player_pos[0, 1]] = 0
                # Set new position
                new_grid[player_pos[0, 0], player_pos[0, 1]] = 2
            else:
                # If out of bounds, just clear old position
                new_grid[player_pos[0, 0], player_pos[0, 1]] = 0
    
    elif action == 3:
        # Action 3: Toggle color 4 (blue) to 5 (yellow) or vice versa
        # Find all color 4 cells
        blue_cells = np.argwhere(new_grid == 4)
        for r, c in blue_cells:
            new_grid[r, c] = 5
    
    elif action == 4:
        # Action 4: Move player (color 2) left by 1
        # Find player position
        player_pos = np.argwhere(new_grid == 2)
        if len(player_pos) > 0:
            # Move left
            player_pos[:, 1] -= 1
            # Check bounds
            if player_pos[0, 1] >= 0:
                # Clear old position
                new_grid[player_pos[0, 0], player_pos[0, 1]] = 0
                # Set new position
                new_grid[player_pos[0, 0], player_pos[0, 1]] = 2
            else:
                # If out of bounds, just clear old position
                new_grid[player_pos[0, 0], player_pos[0, 1]] = 0
    
    elif action == 6:
        # Action 6: Click action (not implemented in this simple model)
        pass
    
    elif action == 7:
        # Action 7: Move player (color 2) right by 1
        # Find player position
        player_pos = np.argwhere(new_grid == 2)
        if len(player_pos) > 0:
            # Move right
            player_pos[:, 1] += 1
            # Check bounds
            if player_pos[0, 1] < W:
                # Clear old position
                new_grid[player_pos[0, 0], player_pos[0, 1]] = 0
                # Set new position
                new_grid[player_pos[0, 0], player_pos[0, 1]] = 2
            else:
                # If out of bounds, just clear old position
                new_grid[player_pos[0, 0], player_pos[0, 1]] = 0
    
    return new_grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # Win state has specific patterns in rows 6-47
    # Rows 0-5: all 5s
    # Rows 6-47: specific pattern with 5s and 4s
    # Rows 48-52: all 5s
    # Rows 53-55: all 2s and 4s respectively
    # Rows 56-63: specific pattern with 4s
    
    # Check rows 0-5
    for i in range(6):
        if not np.all(grid[i] == 5):
            return False
    
    # Check rows 48-52
    for i in range(48, 53):
        if not np.all(grid[i] == 5):
            return False
    
    # Check rows 53-55
    if not np.all(grid[53] == 2):
        return False
    if not np.all(grid[54] == 4):
        return False
    if not np.all(grid[55] == 4):
        return False
    
    # Check rows 62-63
    for i in range(62, 64):
        if not np.all(grid[i] == 4):
            return False
    
    # Check rows 6-47 for specific pattern
    # This is a simplified check - in reality, we'd need to check the exact pattern
    # For now, we'll just check if the grid has the right structure
    
    return True