import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 1:
        # Move all objects of color 3 one step down (increase row index)
        new_grid = grid.copy()
        for r in range(H - 1):
            for c in range(W):
                if grid[r, c] == 3:
                    if grid[r + 1, c] != 4:
                        new_grid[r + 1, c] = 3
                        new_grid[r, c] = 4
                    else:
                        new_grid[r + 1, c] = 4
        return new_grid
    elif action == 3:
        # Move all objects of color 3 one step right (increase col index)
        new_grid = grid.copy()
        for r in range(H):
            for c in range(W - 1):
                if grid[r, c] == 3:
                    if grid[r, c + 1] != 4:
                        new_grid[r, c + 1] = 3
                        new_grid[r, c] = 4
                    else:
                        new_grid[r, c + 1] = 4
        return new_grid
    elif action == 6:
        # Click action - no change
        return grid
    else:
        return grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # Win state has specific structure:
    # - Top rows (0-4) are mostly 4s with some 5s
    # - Middle rows have specific patterns
    # - Bottom rows have specific patterns
    # Based on the win state provided, we check for the specific structure
    
    # Check row 0-4: should be 5x4, 4x60
    for i in range(5):
        if not (grid[i, 0:4].all() == 5 and grid[i, 4:].all() == 4):
            return False
    
    # Check row 5-9: should be 5x4, 4x15, 3x35, 4x10
    for i in range(5, 10):
        if not (grid[i, 0:4].all() == 5 and 
                grid[i, 4:19].all() == 4 and 
                grid[i, 19:54].all() == 3 and 
                grid[i, 54:].all() == 4):
            return False
    
    # Check row 10-14: should be 5x4, 4x5, 3x45, 4x10
    for i in range(10, 15):
        if not (grid[i, 0:4].all() == 5 and 
                grid[i, 4:9].all() == 4 and 
                grid[i, 9:54].all() == 3 and 
                grid[i, 54:].all() == 4):
            return False
    
    # Check row 15-19: should be 5x4, 4x5, 3x15, 4x5, 3x10, 4x5, 3x10, 4x10
    for i in range(15, 20):
        if not (grid[i, 0:4].all() == 5 and 
                grid[i, 4:9].all() == 4 and 
                grid[i, 9:24].all() == 3 and 
                grid[i, 24:29].all() == 4 and 
                grid[i, 29:39].all() == 3 and 
                grid[i, 39:44].all() == 4 and 
                grid[i, 44:54].all() == 3 and 
                grid[i, 54:].all() == 4):
            return False
    
    # Check row 20-24: should be 5x4, 4x5, 3x15, 4x5, 3x10, 4x10, 3x10, 4x5
    for i in range(20, 25):
        if not (grid[i, 0:4].all() == 5 and 
                grid[i, 4:9].all() == 4 and 
                grid[i, 9:24].all() == 3 and 
                grid[i, 24:29].all() == 4 and 
                grid[i, 29:39].all() == 3 and 
                grid[i, 39:49].all() == 4 and 
                grid[i, 49:59].all() == 3 and 
                grid[i, 59:].all() == 4):
            return False
    
    # Check row 25-29: should be 5x4, 4x10, 3x5, 4x15, 3x10, 4x5, 3x10, 4x5
    for i in range(25, 30):
        if not (grid[i, 0:4].all() == 5 and 
                grid[i, 4:14].all() == 4 and 
                grid[i, 14:19].all() == 3 and 
                grid[i, 19:34].all() == 4 and 
                grid[i, 34:44].all() == 3 and 
                grid[i, 44:49].all() == 4 and 
                grid[i, 49:59].all() == 3 and 
                grid[i, 59:].all() == 4):
            return False
    
    # Check row 30-34: should be 5x4, 4x10, 3x5, 4x15, 3x10, 4x5, 3x5, 4x10
    for i in range(30, 35):
        if not (grid[i, 0:4].all() == 5 and 
                grid[i, 4:14].all() == 4 and 
                grid[i, 14:19].all() == 3 and 
                grid[i, 19:34].all() == 4 and 
                grid[i, 34:44].all() == 3 and 
                grid[i, 44:49].all() == 4 and 
                grid[i, 49:54].all() == 3 and 
                grid[i, 54:].all() == 4):
            return False
    
    # Check row 35-39: should be 5x4, 4x10, 3x5, 4x10, 3x10, 4x10, 3x5, 4x10
    for i in range(35, 40):
        if not (grid[i, 0:4].all() == 5 and 
                grid[i, 4:14].all() == 4 and 
                grid[i, 14:24].all() == 3 and 
                grid[i, 24:34].all() == 4 and 
                grid[i, 34:44].all() == 3 and 
                grid[i, 44:54].all() == 4 and 
                grid[i, 54:].all() == 3 and 
                grid[i, 59:].all() == 4):
            return False
    
    # Check row 40-44: should be 5x4, 4x8, 3x1, 5x7, 3x1, 4x8, 12x5, 3x5, 4x5, 3x15, 4x5
    for i in range(40, 45):
        if not (grid[i, 0:4].all() == 5 and 
                grid[i, 4:12].all() == 4 and 
                grid[i, 12:13].all() == 3 and 
                grid[i, 13:20].all() == 5 and 
                grid[i, 20:21].all() == 3 and 
                grid[i, 21:29].all() == 4 and 
                grid[i, 29:41].all() == 5 and 
                grid[i, 41:46].all() == 3 and 
                grid[i, 46:51].all() == 4 and 
                grid[i, 51:66].all() == 3 and 
                grid[i, 66:].all() == 4):
            return False
    
    # Check row 45-49: should be 5x4, 4x8, 3x1, 5x7, 3x1, 4x23, 3x15, 4x5
    for i in range(45, 50):
        if not (grid[i, 0:4].all() == 5 and 
                grid[i, 4:12].all() == 4 and 
                grid[i, 12:13].all() == 3 and 
                grid[i, 13:20].all() == 5 and 
                grid[i, 20:21].all() == 3 and 
                grid[i, 21:44].all() == 4 and 
                grid[i, 44:59].all() == 3 and 
                grid[i, 59:].all() == 4):
            return False
    
    # Check row 50-51: should be 5x4, 4x35, 3x20, 4x5
    for i in range(50, 52):
        if not (grid[i, 0:4].all() == 5 and 
                grid[i, 4:39].all() == 4 and 
                grid[i, 39:59].all() == 3 and 
                grid[i, 59:].all() == 4):
            return False
    
    # Check row 52: should be 4x39, 3x1, 11x1, 3x1, 11x1, 3x16, 4x5
    if not (grid[52, 0:39].all() == 4 and 
            grid[52, 39:40].all() == 3 and 
            grid[52, 40:41].all() == 11 and 
            grid[52, 41:42].all() == 3 and 
            grid[52, 42:43].all() == 11 and 
            grid[52, 43:59].all() == 3 and 
            grid[52, 59:].all() == 4):
        return False
    
    # Check row 53: should be 4x1, 5x10, 4x28, 3x1, 11x3, 3x16, 4x5
    if not (grid[53, 0:1].all() == 4 and 
            grid[53, 1:11].all() == 5 and 
            grid[53, 11:39].all() == 4 and 
            grid[53, 39:40].all() == 3 and 
            grid[53, 40:43].all() == 11 and 
            grid[53, 43:59].all() == 3 and 
            grid[53, 59:].all() == 4):
        return False
    
    # Check row 54: should be 4x1, 5x10, 4x28, 3x20, 4x5
    if not (grid[54, 0:1].all() == 4 and 
            grid[54, 1:11].all() == 5 and 
            grid[54, 11:39].all() == 4 and 
            grid[54, 39:59].all() == 3 and 
            grid[54, 59:].all() == 4):
        return False
    
    # Check row 55-56: should be 4x1, 5x2, 9x6, 5x2, 4x53
    for i in range(55, 57):
        if not (grid[i, 0:1].all() == 4 and 
                grid[i, 1:3].all() == 5 and 
                grid[i, 3:9].all() == 9 and 
                grid[i, 9:11].all() == 5 and 
                grid[i, 11:].all() == 4):
            return False
    
    # Check row 57-58: should be 4x1, 5x6, 9x2, 5x2, 4x53
    for i in range(57, 59):
        if not (grid[i, 0:1].all() == 4 and 
                grid[i, 1:7].all() == 5 and 
                grid[i, 7:9].all() == 9 and 
                grid[i, 9:11].all() == 5 and 
                grid[i, 11:].all() == 4):
            return False
    
    # Check row 59: should be 4x1, 5x2, 9x2, 5x2, 9x2, 5x2, 4x53
    if not (grid[59, 0:1].all() == 4 and 
            grid[59, 1:3].all() == 5 and 
            grid[59, 3:5].all() == 9 and 
            grid[59, 5:7].all() == 5 and 
            grid[59, 7:9].all() == 9 and 
            grid[59, 9:11].all() == 5 and 
            grid[59, 11:].all() == 4):
        return False
    
    # Check row 60: should be 4x1, 5x2, 9x2, 5x2, 9x2, 5x2, 4x1, 5x52
    if not (grid[60, 0:1].all() == 4 and 
            grid[60, 1:3].all() == 5 and 
            grid[60, 3:5].all() == 9 and 
            grid[60, 5:7].all() == 5 and 
            grid[60, 7:9].all() == 9 and 
            grid[60, 9:11].all() == 5 and 
            grid[60, 11:12].all() == 4 and 
            grid[60, 12:].all() == 5):
        return False
    
    # Check row 61-62: should be 4x1, 5x10, 4x1, 5x1, 11x42, 5x1, 8x2, 5x1, 8x2, 5x1, 8x2
    for i in range(61, 63):
        if not (grid[i, 0:1].all() == 4 and 
                grid[i, 1:11].all() == 5 and 
                grid[i, 11:12].all() == 4 and 
                grid[i, 12:13].all() == 5 and 
                grid[i, 13:55].all() == 11 and 
                grid[i, 55:56].all() == 5 and 
                grid[i, 56:58].all() == 8 and 
                grid[i, 58:59].all() == 5 and 
                grid[i, 59:61].all() == 8 and 
                grid[i, 61:62].all() == 5 and 
                grid[i, 62:].all() == 8):
            return False
    
    # Check row 63: should be 4x12, 5x52
    if not (grid[63, 0:12].all() == 4 and 
            grid[63, 12:].all() == 5):
        return False
    
    return True