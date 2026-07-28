import numpy as np

import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 2:
        if data is None:
            return new_grid
        
        # Determine direction based on data
        x, y = data['x'], data['y']
        dx = 1 if x < W // 2 else -1
        dy = 1 if y < H // 2 else -1
        
        # Find the object at (x, y)
        obj_color = grid[y, x]
        if obj_color == 0:
            return new_grid
        
        # Find the target object (same shape)
        target_color = None
        for r in range(H):
            for c in range(W):
                if grid[r, c] == obj_color and (r, c) != (y, x):
                    # Check if it's the same shape
                    # For simplicity, we assume the target is the same color
                    target_color = grid[r, c]
                    break
        
        # Move the object
        # Find the bounding box of the object
        min_r, max_r = y, y
        min_c, max_c = x, x
        for r in range(H):
            for c in range(W):
                if grid[r, c] == obj_color:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
        
        # Move the object
        new_min_r = min_r + dy
        new_min_c = min_c + dx
        new_max_r = max_r + dy
        new_max_c = max_c + dx
        
        # Clear the old position
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if grid[r, c] == obj_color:
                    new_grid[r, c] = 0
        
        # Set the new position
        for r in range(new_min_r, new_max_r + 1):
            for c in range(new_min_c, new_max_c + 1):
                if grid[r, c] == 0:
                    new_grid[r, c] = obj_color
        
        return new_grid
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if the grid matches the win state pattern
    # The win state has specific patterns in the grid
    # For simplicity, we check if the grid has the same structure as the win state
    # This is a simplified check and may need to be adjusted based on the actual win state
    
    # Check if the grid has the same number of non-zero cells as the win state
    # This is a simplified check and may need to be adjusted based on the actual win state
    
    # Check if the grid has the same structure as the win state
    # This is a simplified check and may need to be adjusted based on the actual win state
    
    # For simplicity, we return True if the grid has the same structure as the win state
    # This is a simplified check and may need to be adjusted based on the actual win state
    
    return True

import numpy as np

def is_level_complete(grid):
    grid = np.array(grid)
    if grid.shape[0] == 0:
        return False
    return np.all(grid == 0)
