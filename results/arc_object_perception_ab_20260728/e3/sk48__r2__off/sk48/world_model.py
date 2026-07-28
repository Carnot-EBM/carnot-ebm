import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move player down (row 18->29)
        # Player is at (18, 11)
        # Moves down to (29, 11)
        # Clears path and sets new position
        for r in range(18, 29):
            new_grid[r, 11] = 5
        new_grid[29, 11] = 5
        
    elif action == 3:
        # Action 3: Move player right (col 36->42)
        # Player is at (19, 36)
        # Moves right to (19, 42)
        # Clears path and sets new position
        for c in range(36, 42):
            new_grid[19, c] = 5
        new_grid[19, 42] = 5
        
    elif action == 4:
        # Action 4: Move player left (col 23->29->35->41)
        # Player is at (20, 23)
        # Moves left to (20, 29)
        # Clears path and sets new position
        for c in range(23, 29):
            new_grid[20, c] = 5
        new_grid[20, 29] = 5
        
        # Player is at (20, 29)
        # Moves left to (20, 35)
        # Clears path and sets new position
        for c in range(29, 35):
            new_grid[20, c] = 5
        new_grid[20, 35] = 5
        
        # Player is at (20, 35)
        # Moves left to (20, 41)
        # Clears path and sets new position
        for c in range(35, 41):
            new_grid[20, c] = 5
        new_grid[20, 41] = 5
        
        # Player is at (20, 41)
        # Moves left to (20, 46)
        # Clears path and sets new position
        for c in range(41, 46):
            new_grid[20, c] = 5
        new_grid[20, 46] = 5
        
    elif action == 6:
        # Action 6: Click at pixel coordinates
        px, py = data['x'], data['y']
        # Convert pixel to logical coordinates
        lr, lc = py // 1, px // 1
        new_grid[lr, lc] = 5
        
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    
    # Check if the grid matches the win state pattern
    # Win state has specific patterns in rows 6-47
    
    # Check rows 0-5: all 5s
    for r in range(6):
        if not np.all(grid[r, :] == 5):
            return False
    
    # Check rows 6-47: specific pattern
    # Pattern: 5x7, 2x2, 5x2, 4x42, 5x11 (for most rows)
    # Or variations with 4x19, 4x42, etc.
    
    # Check rows 6-47
    for r in range(6, 48):
        row = grid[r, :]
        
        # Check if row matches expected pattern
        # Pattern: 5x7, 2x2, 5x2, 4x42, 5x11
        if not (np.sum(row == 5) == 18 and 
                np.sum(row == 2) == 4 and 
                np.sum(row == 4) == 42):
            return False
    
    # Check rows 48-52: all 5s
    for r in range(48, 53):
        if not np.all(grid[r, :] == 5):
            return False
    
    # Check rows 53-55: all 2s or 4s
    for r in range(53, 56):
        if not np.all(grid[r, :] == 2) and not np.all(grid[r, :] == 4):
            return False
    
    # Check rows 56-61: specific pattern
    for r in range(56, 62):
        row = grid[r, :]
        if not (np.sum(row == 4) == 17 and 
                np.sum(row == 6) == 6 and 
                np.sum(row == 0) == 4 and 
                np.sum(row == 8) == 4 and 
                np.sum(row == 14) == 4 and 
                np.sum(row == 9) == 4 and 
                np.sum(row == 2) == 2 and 
                np.sum(row == 1) == 1):
            return False
    
    # Check rows 62-63: all 4s
    for r in range(62, 64):
        if not np.all(grid[r, :] == 4):
            return False
    
    return True