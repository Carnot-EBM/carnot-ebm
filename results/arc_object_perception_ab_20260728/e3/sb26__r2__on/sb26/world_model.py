import numpy as np

import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return new_grid
        
        # Determine the 3x3 area around the click
        y_start = max(0, py - 1)
        y_end = min(H, py + 2)
        x_start = max(0, px - 1)
        x_end = min(W, px + 2)
        
        # Create a mask for the 3x3 area
        mask = np.zeros((H, W), dtype=bool)
        mask[y_start:y_end, x_start:x_end] = True
        
        # Apply the toggle pattern: center and cross
        # Pattern: center, up, down, left, right
        # Based on observed deltas, the pattern is a cross shape
        # Center
        if mask[py, px]:
            new_grid[py, px] = 0
        
        # Up
        if py > 0 and mask[py - 1, px]:
            new_grid[py - 1, px] = 0
            
        # Down
        if py < H - 1 and mask[py + 1, px]:
            new_grid[py + 1, px] = 0
            
        # Left
        if px > 0 and mask[py, px - 1]:
            new_grid[py, px - 1] = 0
            
        # Right
        if px < W - 1 and mask[py, px + 1]:
            new_grid[py, px + 1] = 0
            
        return new_grid
    else:
        # Directional actions (1-5)
        # Based on the game structure, these actions move the player
        # but don't change the grid state directly
        return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    
    # Check if the grid matches the win state pattern
    # The win state has specific patterns in the bottom rows
    # Check rows 57-60 for the specific pattern
    if H < 61:
        return False
    
    # Check the pattern in rows 57-60
    # Pattern: 4x9, 8x4, 4x3, 15x4, 4x3, 14x4, 4x3, 12x4, 4x3, 6x4, 4x3, 9x4, 4x3, 11x4, 4x9
    for row in range(57, 61):
        if row >= H:
            return False
        
        # Check if the row matches the expected pattern
        # The pattern is: 4x9, 8x4, 4x3, 15x4, 4x3, 14x4, 4x3, 12x4, 4x3, 6x4, 4x3, 9x4, 4x3, 11x4, 4x9
        expected = [9, 4, 3, 15, 4, 3, 14, 4, 3, 12, 4, 3, 6, 4, 3, 9, 4]
        expected = np.array(expected)
        
        # Check if the row matches the expected pattern
        if not np.array_equal(grid[row], expected):
            return False
    
    # Check if the top rows match the win state pattern
    # Rows 0-7 should have the pattern: 4x7, 5x50, 4x7
    for row in range(8):
        if row >= H:
            return False
        
        # Check if the row matches the expected pattern
        # The pattern is: 4x7, 5x50, 4x7
        expected = np.array([7, 50, 7])
        
        # Check if the row matches the expected pattern
        if not np.array_equal(grid[row], expected):
            return False
    
    return True

def is_level_complete(grid):
    import numpy as np
    grid = np.array(grid)
    return np.all(grid == 0)
