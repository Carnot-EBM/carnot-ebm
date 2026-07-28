import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move Up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if new_grid[r, c] == 13:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c] = 13
                    break
    
    elif action == 2:
        # Action 2: Move Down
        for c in range(W):
            for r in range(H - 1):
                if new_grid[r, c] == 13:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c] = 13
                    break
    
    elif action == 3:
        # Action 3: Move Left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if new_grid[r, c] == 13:
                    new_grid[r, c] = 0
                    new_grid[r, c - 1] = 13
                    break
    
    elif action == 5:
        # Action 5: Move Right
        for r in range(H):
            for c in range(W - 1):
                if new_grid[r, c] == 13:
                    new_grid[r, c] = 0
                    new_grid[r, c + 1] = 13
                    break
    
    elif action == 6:
        # Action 6: Click at pixel coordinates
        px, py = data['x'], data['y']
        # Convert pixel to logical
        r, c = py // 1, px // 1
        if 0 <= r < H and 0 <= c < W:
            new_grid[r, c] = 13
    
    elif action == 7:
        # Action 7: Toggle
        if data is not None:
            px, py = data['x'], data['y']
            r, c = py // 1, px // 1
            if 0 <= r < H and 0 <= c < W:
                if new_grid[r, c] == 13:
                    new_grid[r, c] = 0
                else:
                    new_grid[r, c] = 13
    
    return new_grid

def is_level_complete(grid):
    # Check if the level is complete
    # Based on the observed transitions, the level is complete when the grid is in a win state
    # Since we don't have explicit win state data, we assume the level is complete if the grid matches the win state pattern
    # For now, we return False as a placeholder
    return False