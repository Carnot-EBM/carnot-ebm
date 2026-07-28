import numpy as np

import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 15
    elif action == 3:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 15
            if py > 0:
                new_grid[py-1, px] = 15
            if py < H-1:
                new_grid[py+1, px] = 15
            if px > 0:
                new_grid[py, px-1] = 15
            if px < W-1:
                new_grid[py, px+1] = 15
    elif action == 1:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            if py > 0 and new_grid[py-1, px] != 15:
                new_grid[py-1, px] = 15
            if py < H-1 and new_grid[py+1, px] != 15:
                new_grid[py+1, px] = 15
            if px > 0 and new_grid[py, px-1] != 15:
                new_grid[py, px-1] = 15
            if px < W-1 and new_grid[py, px+1] != 15:
                new_grid[py, px+1] = 15
    elif action == 2:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            if py > 0 and new_grid[py-1, px] != 15:
                new_grid[py-1, px] = 15
            if py < H-1 and new_grid[py+1, px] != 15:
                new_grid[py+1, px] = 15
            if px > 0 and new_grid[py, px-1] != 15:
                new_grid[py, px-1] = 15
            if px < W-1 and new_grid[py, px+1] != 15:
                new_grid[py, px+1] = 15
    elif action == 4:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            if py > 0 and new_grid[py-1, px] != 15:
                new_grid[py-1, px] = 15
            if py < H-1 and new_grid[py+1, px] != 15:
                new_grid[py+1, px] = 15
            if px > 0 and new_grid[py, px-1] != 15:
                new_grid[py, px-1] = 15
            if px < W-1 and new_grid[py, px+1] != 15:
                new_grid[py, px+1] = 15
    elif action == 5:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            if py > 0 and new_grid[py-1, px] != 15:
                new_grid[py-1, px] = 15
            if py < H-1 and new_grid[py+1, px] != 15:
                new_grid[py+1, px] = 15
            if px > 0 and new_grid[py, px-1] != 15:
                new_grid[py, px-1] = 15
            if px < W-1 and new_grid[py, px+1] != 15:
                new_grid[py, px+1] = 15
    elif action == 7:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            if py > 0 and new_grid[py-1, px] != 15:
                new_grid[py-1, px] = 15
            if py < H-1 and new_grid[py+1, px] != 15:
                new_grid[py+1, px] = 15
            if px > 0 and new_grid[py, px-1] != 15:
                new_grid[py, px-1] = 15
            if px < W-1 and new_grid[py, px+1] != 15:
                new_grid[py, px+1] = 15
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for y in range(H):
        for x in range(W):
            if grid[y, x] != 5 and grid[y, x] != 10 and grid[y, x] != 14 and grid[y, x] != 15 and grid[y, x] != 0:
                return False
    return True

def is_level_complete(grid):
    import numpy as np
    grid = np.array(grid)
    if grid.shape != (10, 10):
        return False
    return np.all(grid == 0)
