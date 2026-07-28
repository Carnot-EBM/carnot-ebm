import numpy as np

import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 0:
        if data is None:
            return new_grid
        
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return new_grid
            
        new_grid[py, px] = 14
        
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 14:
                    new_grid[r, c] = 3
        
        return new_grid

    elif action == 1:
        if data is None:
            return new_grid
        
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return new_grid
            
        new_grid[py, px] = 14
        
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 14:
                    new_grid[r, c] = 3
        
        return new_grid

    elif action == 2:
        if data is None:
            return new_grid
        
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return new_grid
            
        new_grid[py, px] = 14
        
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 14:
                    new_grid[r, c] = 3
        
        return new_grid

    elif action == 3:
        if data is None:
            return new_grid
        
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return new_grid
            
        new_grid[py, px] = 14
        
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 14:
                    new_grid[r, c] = 3
        
        return new_grid

    elif action == 4:
        if data is None:
            return new_grid
        
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return new_grid
            
        new_grid[py, px] = 14
        
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 14:
                    new_grid[r, c] = 3
        
        return new_grid

    elif action == 5:
        if data is None:
            return new_grid
        
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return new_grid
            
        new_grid[py, px] = 14
        
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 14:
                    new_grid[r, c] = 3
        
        return new_grid

    elif action == 6:
        if data is None:
            return new_grid
        
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return new_grid
            
        new_grid[py, px] = 14
        
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 14:
                    new_grid[r, c] = 3
        
        return new_grid

    elif action == 7:
        if data is None:
            return new_grid
        
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return new_grid
            
        new_grid[py, px] = 14
        
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 14:
                    new_grid[r, c] = 3
        
        return new_grid

    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 14:
                return False
            if grid[r, c] == 3:
                return False
    
    return True

def is_level_complete(grid):
    import numpy as np
    g = np.array(grid)
    return np.all(g == 0)
