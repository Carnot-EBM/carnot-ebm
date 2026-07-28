import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move player down
        # Find player (color 5)
        player_pos = np.argwhere(new_grid == 5)
        if len(player_pos) == 0:
            return new_grid
            
        # Check if player is at bottom
        if player_pos[-1, 0] == H - 1:
            return new_grid
            
        # Move player down
        new_grid[player_pos[-1, 0] + 1, player_pos[-1, 1]] = 5
        new_grid[player_pos[-1, 0], player_pos[-1, 1]] = 0
        
        # Apply gravity to blocks above player
        for r in range(player_pos[-1, 0], -1, -1):
            for c in range(W):
                if new_grid[r, c] != 0 and new_grid[r, c] != 5:
                    # Move block down
                    new_grid[r + 1, c] = new_grid[r, c]
                    new_grid[r, c] = 0
                    break
    
    elif action == 3:
        # Action 3: Move player up
        # Find player (color 5)
        player_pos = np.argwhere(new_grid == 5)
        if len(player_pos) == 0:
            return new_grid
            
        # Check if player is at top
        if player_pos[0, 0] == 0:
            return new_grid
            
        # Move player up
        new_grid[player_pos[0, 0] - 1, player_pos[0, 1]] = 5
        new_grid[player_pos[0, 0], player_pos[0, 1]] = 0
        
        # Apply gravity to blocks below player
        for r in range(player_pos[0, 0], H):
            for c in range(W):
                if new_grid[r, c] != 0 and new_grid[r, c] != 5:
                    # Move block up
                    new_grid[r - 1, c] = new_grid[r, c]
                    new_grid[r, c] = 0
                    break
    
    elif action == 2:
        # Action 2: Move player left
        # Find player (color 5)
        player_pos = np.argwhere(new_grid == 5)
        if len(player_pos) == 0:
            return new_grid
            
        # Check if player is at left
        if player_pos[0, 1] == 0:
            return new_grid
            
        # Move player left
        new_grid[player_pos[0, 0], player_pos[0, 1] - 1] = 5
        new_grid[player_pos[0, 0], player_pos[0, 1]] = 0
        
        # Apply gravity to blocks right of player
        for c in range(player_pos[0, 1], W):
            for r in range(H):
                if new_grid[r, c] != 0 and new_grid[r, c] != 5:
                    # Move block left
                    new_grid[r, c - 1] = new_grid[r, c]
                    new_grid[r, c] = 0
                    break
    
    elif action == 4:
        # Action 4: Move player right
        # Find player (color 5)
        player_pos = np.argwhere(new_grid == 5)
        if len(player_pos) == 0:
            return new_grid
            
        # Check if player is at right
        if player_pos[0, 1] == W - 1:
            return new_grid
            
        # Move player right
        new_grid[player_pos[0, 0], player_pos[0, 1] + 1] = 5
        new_grid[player_pos[0, 0], player_pos[0, 1]] = 0
        
        # Apply gravity to blocks left of player
        for c in range(player_pos[0, 1], -1, -1):
            for r in range(H):
                if new_grid[r, c] != 0 and new_grid[r, c] != 5:
                    # Move block right
                    new_grid[r, c + 1] = new_grid[r, c]
                    new_grid[r, c] = 0
                    break
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    
    # Check if all cells are filled with color 6 or 15
    for r in range(H):
        for c in range(W):
            if grid[r, c] not in [6, 15]:
                return False
    
    return True