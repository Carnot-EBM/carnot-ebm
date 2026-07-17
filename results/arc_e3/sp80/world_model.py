import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 4:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            if grid[py, px] == 12:
                new_grid[py, px] = 0
    elif action == 5:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            if grid[py, px] == 12:
                new_grid[py, px] = 0
    elif action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            if grid[py, px] == 12:
                new_grid[py, px] = 0
    else:
        return new_grid
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    
    for r in range(64):
        row_str = ""
        for c in range(64):
            val = grid[r, c]
            if val == 12:
                row_str += "12x1"
            elif val == 0:
                row_str += "0x1"
            else:
                row_str += f"{val}x1"
        
        if row_str != "r" + str(r) + ":" + row_str:
            return False
    
    return True