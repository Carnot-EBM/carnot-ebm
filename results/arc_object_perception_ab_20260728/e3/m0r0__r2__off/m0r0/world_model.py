import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Convert pixel to logical
        lx, ly = px // 1, py // 1
        # Check bounds
        if ly < 0 or ly >= H or lx < 0 or lx >= W:
            return new_grid
            
        # Find the color at the clicked position
        current_color = new_grid[ly, lx]
        
        # If the clicked cell is 0 (empty), fill it with 5
        if current_color == 0:
            new_grid[ly, lx] = 5
        # If the clicked cell is 5, toggle it to 0
        elif current_color == 5:
            new_grid[ly, lx] = 0
            
    elif action == 3:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        lx, ly = px // 1, py // 1
        if ly < 0 or ly >= H or lx < 0 or lx >= W:
            return new_grid
            
        # Toggle the clicked cell
        current_color = new_grid[ly, lx]
        if current_color == 5:
            new_grid[ly, lx] = 0
        elif current_color == 0:
            new_grid[ly, lx] = 5
            
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    
    # Check if the grid matches the win state pattern
    # The win state has specific patterns in the rows
    
    # Check row 0
    if not (np.sum(grid[0] == 5) == 64):
        return False
        
    # Check row 63
    if not (np.sum(grid[63] == 5) == 64):
        return False
        
    # Check middle rows (1-5, 46-53)
    for i in range(1, 6):
        if not (np.sum(grid[i] == 6) == 32 and np.sum(grid[i] == 15) == 32):
            return False
    for i in range(46, 54):
        if not (np.sum(grid[i] == 6) == 32 and np.sum(grid[i] == 15) == 32):
            return False
            
    # Check rows 6-9, 14-17, 42-45
    for i in range(6, 10):
        if not (np.sum(grid[i] == 6) == 6 and np.sum(grid[i] == 5) == 48 and np.sum(grid[i] == 15) == 6):
            return False
    for i in range(14, 18):
        if not (np.sum(grid[i] == 6) == 6 and np.sum(grid[i] == 5) == 48 and np.sum(grid[i] == 15) == 6):
            return False
    for i in range(42, 46):
        if not (np.sum(grid[i] == 6) == 6 and np.sum(grid[i] == 5) == 48 and np.sum(grid[i] == 15) == 6):
            return False
            
    # Check rows 10-13, 18-21, 26-37, 54-57
    for i in range(10, 14):
        if not (np.sum(grid[i] == 6) == 6 and np.sum(grid[i] == 5) == 32 and np.sum(grid[i] == 10) == 8 and np.sum(grid[i] == 15) == 6):
            return False
    for i in range(18, 22):
        if not (np.sum(grid[i] == 6) == 6 and np.sum(grid[i] == 5) == 8 and np.sum(grid[i] == 6) == 18 and np.sum(grid[i] == 15) == 18):
            return False
    for i in range(26, 38):
        if not (np.sum(grid[i] == 6) == 6 and np.sum(grid[i] == 5) == 20 and np.sum(grid[i] == 8) == 1 and np.sum(grid[i] == 15) == 6):
            return False
    for i in range(54, 58):
        if not (np.sum(grid[i] == 6) == 6 and np.sum(grid[i] == 5) == 1 and np.sum(grid[i] == 8) == 1 and np.sum(grid[i] == 15) == 6):
            return False
            
    # Check rows 22-25, 28-31, 34-37
    for i in range(22, 26):
        if not (np.sum(grid[i] == 6) == 6 and np.sum(grid[i] == 5) == 8 and np.sum(grid[i] == 6) == 4 and np.sum(grid[i] == 15) == 18):
            return False
    for i in range(28, 32):
        if not (np.sum(grid[i] == 6) == 6 and np.sum(grid[i] == 5) == 20 and np.sum(grid[i] == 8) == 1 and np.sum(grid[i] == 15) == 6):
            return False
    for i in range(34, 38):
        if not (np.sum(grid[i] == 6) == 6 and np.sum(grid[i] == 5) == 20 and np.sum(grid[i] == 8) == 1 and np.sum(grid[i] == 15) == 6):
            return False
            
    # Check rows 38-41
    for i in range(38, 42):
        if not (np.sum(grid[i] == 6) == 6 and np.sum(grid[i] == 5) == 1 and np.sum(grid[i] == 8) == 1 and np.sum(grid[i] == 15) == 6):
            return False
            
    return True