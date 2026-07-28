import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    
    # Action 6 is a click at pixel coordinates
    if action == 6:
        px, py = data['x'], data['y']
        # Convert pixel to logical
        cx, cy = px // 1, py // 1
        if not (0 <= cy < H and 0 <= cx < W):
            return grid.copy()
        
        # The clicked cell is always color 4 (the player)
        # Apply the transformation:
        # 1. Set the clicked cell to 7 (gold)
        # 2. Move all 3-colored cells in the clicked column UP
        # 3. Move all 0-colored cells in the clicked column DOWN
        
        # Create a new grid
        new_grid = grid.copy()
        
        # Step 1: Set clicked cell to 7
        new_grid[cy, cx] = 7
        
        # Step 2: Move 3s up
        # Find all 3s in the column
        col_3s = []
        for r in range(H):
            if new_grid[r, cx] == 3:
                col_3s.append(r)
        
        # Move them up starting from the top
        new_pos = 0
        for r in col_3s:
            new_grid[new_pos, cx] = 3
            new_pos += 1
        
        # Clear the old positions
        for r in col_3s:
            new_grid[r, cx] = 0
        
        # Step 3: Move 0s down
        # Find all 0s in the column
        col_0s = []
        for r in range(H):
            if new_grid[r, cx] == 0:
                col_0s.append(r)
        
        # Move them down starting from the bottom
        new_pos = H - 1
        for r in col_0s:
            new_grid[new_pos, cx] = 0
            new_pos -= 1
        
        # Clear the old positions
        for r in col_0s:
            new_grid[r, cx] = 3
        
        return new_grid
    
    return grid.copy()

def is_level_complete(grid):
    H, W = grid.shape
    
    # Check if the grid matches the win state pattern
    # The win state has specific patterns of colors
    
    # Check row 0: should be all 7s
    if not np.all(grid[0, :] == 7):
        return False
    
    # Check rows 1-15: should be 0s then 3s
    for r in range(1, 16):
        if not np.all(grid[r, :52] == 0) or not np.all(grid[r, 52:] == 3):
            return False
    
    # Check rows 16-19: should be 9s then 0s then 3s
    for r in range(16, 20):
        if not np.all(grid[r, :4] == 9) or not np.all(grid[r, 4:48] == 0) or not np.all(grid[r, 48:] == 3):
            return False
    
    # Check rows 20-23: should be 5s then 3s
    for r in range(20, 24):
        if not np.all(grid[r, :56] == 5) or not np.all(grid[r, 56:] == 3):
            return False
    
    # Check rows 24-27: should be 9s then 0s then 3s
    for r in range(24, 28):
        if not np.all(grid[r, :4] == 9) or not np.all(grid[r, 4:8] == 0) or not np.all(grid[r, 8:] == 3):
            return False
    
    # Check rows 28-31: should be 0s then 3s
    for r in range(28, 32):
        if not np.all(grid[r, :12] == 0) or not np.all(grid[r, 12:] == 3):
            return False
    
    # Check rows 32-35: should be 0s then 3s
    for r in range(32, 36):
        if not np.all(grid[r, :12] == 0) or not np.all(grid[r, 12:] == 3):
            return False
    
    # Check rows 36-39: should be 9s then 0s then 3s
    for r in range(36, 40):
        if not np.all(grid[r, :4] == 9) or not np.all(grid[r, 4:8] == 0) or not np.all(grid[r, 8:] == 3):
            return False
    
    # Check rows 40-43: should be 5s then 2s then 5s then 3s
    for r in range(40, 44):
        if not np.all(grid[r, :28] == 5) or not np.all(grid[r, 28:42] == 2) or not np.all(grid[r, 42:56] == 5) or not np.all(grid[r, 56:] == 3):
            return False
    
    # Check rows 44-47: should be 9s then 0s then 3s
    for r in range(44, 48):
        if not np.all(grid[r, :4] == 9) or not np.all(grid[r, 4:4] == 0) or not np.all(grid[r, 4:] == 3):
            return False
    
    # Check rows 48-51: should be 0s then 3s
    for r in range(48, 52):
        if not np.all(grid[r, :8] == 0) or not np.all(grid[r, 8:] == 3):
            return False
    
    # Check rows 52-53: should be 0s then 2s then 4s then 3s
    for r in range(52, 54):
        if not np.all(grid[r, :8] == 0) or not np.all(grid[r, 8:22] == 2) or not np.all(grid[r, 22:26] == 4) or not np.all(grid[r, 26:] == 3):
            return False
    
    # Check rows 54-55: should be 0s then 2s then 4s then 3s
    for r in range(54, 56):
        if not np.all(grid[r, :8] == 0) or not np.all(grid[r, 8:22] == 2) or not np.all(grid[r, 22:26] == 4) or not np.all(grid[r, 26:] == 3):
            return False
    
    # Check rows 56-57: should be 0s then 2s then 4s then 3s
    for r in range(56, 58):
        if not np.all(grid[r, :8] == 0) or not np.all(grid[r, 8:22] == 2) or not np.all(grid[r, 22:26] == 4) or not np.all(grid[r, 26:] == 3):
            return False
    
    # Check rows 58-63: should be 0s then 3s
    for r in range(58, 64):
        if not np.all(grid[r, :8] == 0) or not np.all(grid[r, 8:] == 3):
            return False
    
    return True