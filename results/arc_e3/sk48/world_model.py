import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move down (gravity)
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] == 5 and grid[r - 1, c] == 0:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c] = 5
    elif action == 2:
        # Action 2: Move up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] == 5 and grid[r - 1, c] == 0:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c] = 5
    elif action == 3:
        # Action 3: Move left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if grid[r, c] == 5 and grid[r, c - 1] == 0:
                    new_grid[r, c] = 0
                    new_grid[r, c - 1] = 5
    elif action == 4:
        # Action 4: Move right
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if grid[r, c] == 5 and grid[r, c + 1] == 0:
                    new_grid[r, c] = 0
                    new_grid[r, c + 1] = 5
    elif action == 5:
        # Action 5: Toggle 0/5
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 0:
                    new_grid[r, c] = 5
                elif grid[r, c] == 5:
                    new_grid[r, c] = 0
    elif action == 6:
        # Action 6: Click at data position
        if data:
            px, py = data['x'], data['y']
            if 0 <= py < H and 0 <= px < W:
                new_grid[py, px] = 5
    elif action == 7:
        # Action 7: Toggle 0/4
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 0:
                    new_grid[r, c] = 4
                elif grid[r, c] == 4:
                    new_grid[r, c] = 0
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    
    # Check if all 5s are in the top rows and all 4s are in the bottom rows
    # This is based on the win state pattern observed
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 5 and r > 6:
                return False
            if grid[r, c] == 4 and r < 56:
                return False
    
    return True