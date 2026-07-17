import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    
    if action == 1:
        # Action 1: Move player down (gravity)
        # Find player (color 5) and move them down
        player_mask = (grid == 5)
        player_indices = np.argwhere(player_mask)
        
        if len(player_indices) == 0:
            return grid
            
        # Move player down by 1 if possible
        moved = False
        for r, c in player_indices:
            if r < H - 1 and grid[r + 1, c] == 0:
                grid[r, c] = 0
                grid[r + 1, c] = 5
                moved = True
        
        return grid
    
    elif action == 2:
        # Action 2: Move player up
        player_mask = (grid == 5)
        player_indices = np.argwhere(player_mask)
        
        if len(player_indices) == 0:
            return grid
            
        moved = False
        for r, c in player_indices:
            if r > 0 and grid[r - 1, c] == 0:
                grid[r, c] = 0
                grid[r - 1, c] = 5
                moved = True
        
        return grid
        
    elif action == 3:
        # Action 3: Move player left
        player_mask = (grid == 5)
        player_indices = np.argwhere(player_mask)
        
        if len(player_indices) == 0:
            return grid
            
        moved = False
        for r, c in player_indices:
            if c > 0 and grid[r, c - 1] == 0:
                grid[r, c] = 0
                grid[r, c - 1] = 5
                moved = True
        
        return grid
        
    elif action == 4:
        # Action 4: Move player right
        player_mask = (grid == 5)
        player_indices = np.argwhere(player_mask)
        
        if len(player_indices) == 0:
            return grid
            
        moved = False
        for r, c in player_indices:
            if c < W - 1 and grid[r, c + 1] == 0:
                grid[r, c] = 0
                grid[r, c + 1] = 5
                moved = True
        
        return grid
        
    elif action == 5:
        # Action 5: Toggle blocks (0 <-> 10)
        player_mask = (grid == 5)
        player_indices = np.argwhere(player_mask)
        
        if len(player_indices) == 0:
            return grid
            
        for r, c in player_indices:
            if grid[r, c] == 10:
                grid[r, c] = 0
            elif grid[r, c] == 0:
                grid[r, c] = 10
        
        return grid
        
    elif action == 6:
        # Action 6: Click action - toggle blocks at clicked position
        if data is None:
            return grid
            
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            if grid[py, px] == 10:
                grid[py, px] = 0
            elif grid[py, px] == 0:
                grid[py, px] = 10
        
        return grid
        
    elif action == 7:
        # Action 7: Collect blocks (remove color 10)
        player_mask = (grid == 5)
        player_indices = np.argwhere(player_mask)
        
        if len(player_indices) == 0:
            return grid
            
        for r, c in player_indices:
            if grid[r, c] == 10:
                grid[r, c] = 0
        
        return grid
    
    return grid

def is_level_complete(grid):
    # Check if all cells are filled with color 6 or 15
    # Or if the grid matches the win state pattern
    unique_colors = np.unique(grid)
    return len(unique_colors) == 2 and 6 in unique_colors and 15 in unique_colors