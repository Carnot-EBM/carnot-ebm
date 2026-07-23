import numpy as np

def engine(grid, action, data):
    grid = grid.copy()
    if action == 1:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 8, py // 8
        if logical_x < 0 or logical_x >= 8 or logical_y < 0 or logical_y >= 8:
            return grid
        
        if grid[logical_y, logical_x] == 0:
            grid[logical_y, logical_x] = 1
        else:
            grid[logical_y, logical_x] = 0
    elif action == 2:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 8, py // 8
        if logical_x < 0 or logical_x >= 8 or logical_y < 0 or logical_y >= 8:
            return grid
        
        if grid[logical_y, logical_x] == 0:
            grid[logical_y, logical_x] = 2
        else:
            grid[logical_y, logical_x] = 0
    elif action == 3:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 8, py // 8
        if logical_x < 0 or logical_x >= 8 or logical_y < 0 or logical_y >= 8:
            return grid
        
        if grid[logical_y, logical_x] == 0:
            grid[logical_y, logical_x] = 3
        else:
            grid[logical_y, logical_x] = 0
    elif action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 8, py // 8
        if logical_x < 0 or logical_x >= 8 or logical_y < 0 or logical_y >= 8:
            return grid
        
        if grid[logical_y, logical_x] == 0:
            grid[logical_y, logical_x] = 4
        else:
            grid[logical_y, logical_x] = 0
    elif action == 5:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 8, py // 8
        if logical_x < 0 or logical_x >= 8 or logical_y < 0 or logical_y >= 8:
            return grid
        
        if grid[logical_y, logical_x] == 0:
            grid[logical_y, logical_x] = 5
        else:
            grid[logical_y, logical_x] = 0
    elif action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 8, py // 8
        if logical_x < 0 or logical_x >= 8 or logical_y < 0 or logical_y >= 8:
            return grid
        
        if grid[logical_y, logical_x] == 0:
            grid[logical_y, logical_x] = 6
        else:
            grid[logical_y, logical_x] = 0
    elif action == 7:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 8, py // 8
        if logical_x < 0 or logical_x >= 8 or logical_y < 0 or logical_y >= 8:
            return grid
        
        if grid[logical_y, logical_x] == 0:
            grid[logical_y, logical_x] = 7
        else:
            grid[logical_y, logical_x] = 0
    return grid

def is_level_complete(grid):
    return False