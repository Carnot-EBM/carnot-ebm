import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        logical_x, logical_y = px, py
        if logical_x < 0 or logical_x >= 64 or logical_y < 0 or logical_y >= 64:
            return grid
        if grid[logical_y, logical_x] == 5:
            return grid
        
        new_grid = grid.copy()
        new_grid[logical_y, logical_x] = 14
        
        r9 = new_grid[9, :]
        r10 = new_grid[10, :]
        r11 = new_grid[11, :]
        r63 = new_grid[63, :]
        
        if logical_x >= 36 and logical_x <= 38:
            r9[logical_x] = 14
            r10[logical_x] = 14
            r11[logical_x] = 14
            r63[logical_x] = 14
        elif logical_x >= 39 and logical_x <= 41:
            r9[logical_x] = 14
            r10[logical_x] = 14
            r11[logical_x] = 14
            r63[logical_x] = 14
        elif logical_x >= 42 and logical_x <= 44:
            r9[logical_x] = 14
            r10[logical_x] = 14
            r11[logical_x] = 14
            r63[logical_x] = 14
        elif logical_x >= 45 and logical_x <= 47:
            r9[logical_x] = 14
            r10[logical_x] = 14
            r11[logical_x] = 14
            r63[logical_x] = 14
        elif logical_x >= 48 and logical_x <= 50:
            r9[logical_x] = 14
            r10[logical_x] = 14
            r11[logical_x] = 14
            r63[logical_x] = 14
        elif logical_x >= 51 and logical_x <= 53:
            r9[logical_x] = 14
            r10[logical_x] = 14
            r11[logical_x] = 14
            r63[logical_x] = 14
        elif logical_x >= 54 and logical_x <= 56:
            r9[logical_x] = 14
            r10[logical_x] = 14
            r11[logical_x] = 14
            r63[logical_x] = 14
        elif logical_x >= 57 and logical_x <= 59:
            r9[logical_x] = 14
            r10[logical_x] = 14
            r11[logical_x] = 14
            r63[logical_x] = 14
        elif logical_x >= 60 and logical_x <= 62:
            r9[logical_x] = 14
            r10[logical_x] = 14
            r11[logical_x] = 14
            r63[logical_x] = 14
            
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    if grid is None:
        return False
    return (grid[9, 33:58] == 15).all() and (grid[10, 33:58] == 15).all() and (grid[11, 33:58] == 15).all()