import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if not (0 <= py < grid.shape[0] and 0 <= px < grid.shape[1]):
            return grid
        if grid[py, px] == 14:
            grid[py, px] = 5
            return grid
        return grid
    
    if action in [1, 2, 3, 4, 5, 7]:
        if action == 1:
            dr, dc = -1, 0
        elif action == 2:
            dr, dc = 0, -1
        elif action == 3:
            dr, dc = 0, 1
        elif action == 4:
            dr, dc = 1, 0
        elif action == 5:
            dr, dc = -1, -1
        elif action == 7:
            dr, dc = -1, 1
        
        if data is None:
            return grid
        
        # Find player position
        player_pos = np.argwhere(grid == 14)
        if len(player_pos) == 0:
            return grid
        
        py, px = player_pos[0]
        
        # Move player
        new_py = py + dr
        new_px = px + dc
        
        # Check bounds
        if not (0 <= new_py < grid.shape[0] and 0 <= new_px < grid.shape[1]):
            return grid
        
        # Check collision
        if grid[new_py, new_px] != 5:
            grid[new_py, new_px] = 14
            grid[py, px] = 5
            return grid
        
        # Push logic
        if grid[new_py, new_px] == 5:
            # Move player
            grid[new_py, new_px] = 14
            grid[py, px] = 5
            
            # Push block
            push_py = new_py + dr
            push_px = new_px + dc
            
            if not (0 <= push_py < grid.shape[0] and 0 <= push_px < grid.shape[1]):
                grid[py, px] = 5
                return grid
            
            if grid[push_py, push_px] == 5:
                grid[py, px] = 5
                return grid
            
            # Move block
            grid[push_py, push_px] = 5
            grid[new_py, new_px] = 5
            
            return grid
    
    return grid

def is_level_complete(grid):
    for i in range(grid.shape[0]):
        if np.all(grid[i] == 5):
            return True
    return False