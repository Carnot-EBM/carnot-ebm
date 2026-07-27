import numpy as np

def engine(grid, action, data):
    if action == 1:
        # Move player down
        h, w = grid.shape
        py = grid[0, 0]
        px = grid[0, 0]
        # Find player position
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 4:
                    px, py = c, r
                    break
            if grid[r, c] == 4:
                break
        
        # Move down
        if py + 1 < h:
            if grid[py + 1, px] == 14:
                grid[py + 1, px] = 4
                grid[py, px] = 14
            else:
                grid[py + 1, px] = 4
                grid[py, px] = 0
        return grid
    
    elif action == 2:
        # Move player up
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 4:
                    px, py = c, r
                    break
            if grid[r, c] == 4:
                break
        
        if py - 1 >= 0:
            if grid[py - 1, px] == 14:
                grid[py - 1, px] = 4
                grid[py, px] = 14
            else:
                grid[py - 1, px] = 4
                grid[py, px] = 0
        return grid
    
    elif action == 3:
        # Move player left
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 4:
                    px, py = c, r
                    break
            if grid[r, c] == 4:
                break
        
        if px - 1 >= 0:
            if grid[py, px - 1] == 14:
                grid[py, px - 1] = 4
                grid[py, px] = 14
            else:
                grid[py, px - 1] = 4
                grid[py, px] = 0
        return grid
    
    elif action == 4:
        # Move player right
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 4:
                    px, py = c, r
                    break
            if grid[r, c] == 4:
                break
        
        if px + 1 < w:
            if grid[py, px + 1] == 14:
                grid[py, px + 1] = 4
                grid[py, px] = 14
            else:
                grid[py, px + 1] = 4
                grid[py, px] = 0
        return grid
    
    elif action == 5:
        # Toggle 14 to 0
        grid = grid.copy()
        grid[grid == 14] = 0
        return grid
    
    elif action == 6:
        # Click action - not implemented in this simplified model
        return grid
    
    elif action == 7:
        # Toggle 14 to 0
        grid = grid.copy()
        grid[grid == 14] = 0
        return grid
    
    return grid

def is_level_complete(grid):
    # Check if all 14s are collected
    return not np.any(grid == 14)