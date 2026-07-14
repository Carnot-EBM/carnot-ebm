import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move Right
        # Find the player (value 1) and move them right if possible
        player_pos = np.argwhere(new_grid == 1)
        if len(player_pos) > 0:
            # Move all players to the right
            for i in range(len(player_pos)):
                r, c = player_pos[i]
                if c < W - 1:
                    new_grid[r, c] = 0
                    new_grid[r, c + 1] = 1
            # Remove any players that moved off the grid (shouldn't happen with right move)
            new_grid = new_grid[:, :W]
    elif action == 2:
        # Action 2: Move Left
        player_pos = np.argwhere(new_grid == 1)
        if len(player_pos) > 0:
            for i in range(len(player_pos)):
                r, c = player_pos[i]
                if c > 0:
                    new_grid[r, c] = 0
                    new_grid[r, c - 1] = 1
    elif action == 3:
        # Action 3: Move Down
        player_pos = np.argwhere(new_grid == 1)
        if len(player_pos) > 0:
            for i in range(len(player_pos)):
                r, c = player_pos[i]
                if r < H - 1:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c] = 1
    elif action == 4:
        # Action 4: Move Up
        player_pos = np.argwhere(new_grid == 1)
        if len(player_pos) > 0:
            for i in range(len(player_pos)):
                r, c = player_pos[i]
                if r > 0:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c] = 1
    elif action == 5:
        # Action 5: Move Diagonal Down-Right
        player_pos = np.argwhere(new_grid == 1)
        if len(player_pos) > 0:
            for i in range(len(player_pos)):
                r, c = player_pos[i]
                if r < H - 1 and c < W - 1:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c + 1] = 1
    elif action == 6:
        # Action 6: Click (data contains x, y in pixel coordinates)
        if data is not None:
            px, py = data['x'], data['y']
            # Convert pixel to logical coordinates
            r = py // 8
            c = px // 8
            if 0 <= r < H and 0 <= c < W:
                # Toggle the cell at the clicked position
                new_grid[r, c] = 1 - new_grid[r, c]
    elif action == 7:
        # Action 7: Move Diagonal Up-Left
        player_pos = np.argwhere(new_grid == 1)
        if len(player_pos) > 0:
            for i in range(len(player_pos)):
                r, c = player_pos[i]
                if r > 0 and c > 0:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c - 1] = 1
    
    return new_grid

def is_level_complete(grid):
    # Check if the level is complete
    # Assuming the level is complete when the player (value 1) reaches the bottom-right corner
    H, W = grid.shape
    return grid[H-1, W-1] == 1