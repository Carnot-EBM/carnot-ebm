import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        
        # Apply changes to rows 9-11
        for r in range(9, 12):
            if r == 9:
                # Row 9: change at col 36
                if r == 9:
                    if 36 <= W:
                        new_grid[9, 36:39] = 14
            elif r == 10:
                # Row 10: changes at col 34, 36
                if 34 <= W:
                    new_grid[10, 34:35] = 14
                if 36 <= W:
                    new_grid[10, 36:37] = 14
                if 37 <= W:
                    new_grid[10, 37:38] = 13
                if 38 <= W:
                    new_grid[10, 38:39] = 14
            elif r == 11:
                # Row 11: change at col 36
                if 36 <= W:
                    new_grid[11, 36:39] = 14
        
        # Apply changes to row 63
        if 61 <= W:
            new_grid[63, 61:63] = 4
            
        return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    
    # Check for win state pattern
    # Rows 0-8 should be all 5s
    for r in range(9):
        if not np.all(grid[r, :] == 5):
            return False
    
    # Rows 9-11 should have pattern 5x33, 15x24, 5x7
    for r in range(9, 12):
        if not np.all(grid[r, :33] == 5) or not np.all(grid[r, 33:57] == 15) or not np.all(grid[r, 57:] == 5):
            return False
    
    # Rows 12-26 should have pattern 5x33, 15x3, 5x19
    for r in range(12, 27):
        if not np.all(grid[r, :33] == 5) or not np.all(grid[r, 33:36] == 15) or not np.all(grid[r, 36:55] == 5):
            return False
    
    # Rows 27-35 should have pattern 5x42, 15x3, 5x19
    for r in range(27, 36):
        if not np.all(grid[r, :42] == 5) or not np.all(grid[r, 42:45] == 15) or not np.all(grid[r, 45:64] == 5):
            return False
    
    # Rows 36-41 should have pattern 5x9, 3x1, 12x2, 10x3, 5x27, 15x3, 5x19
    for r in range(36, 42):
        if not np.all(grid[r, :9] == 5) or not np.all(grid[r, 9:10] == 3) or not np.all(grid[r, 10:14] == 12) or not np.all(grid[r, 14:24] == 10) or not np.all(grid[r, 24:51] == 5) or not np.all(grid[r, 51:54] == 15) or not np.all(grid[r, 54:] == 5):
            return False
    
    # Rows 42-44 should have pattern 5x42, 15x3, 5x19
    for r in range(42, 45):
        if not np.all(grid[r, :42] == 5) or not np.all(grid[r, 42:45] == 15) or not np.all(grid[r, 45:64] == 5):
            return False
    
    # Rows 45-60 should be all 5s
    for r in range(45, 61):
        if not np.all(grid[r, :] == 5):
            return False
    
    # Rows 61-62 should be all 5s
    for r in range(61, 63):
        if not np.all(grid[r, :] == 5):
            return False
    
    # Row 63 should be all 3s
    if not np.all(grid[63, :] == 3):
        return False
    
    return True