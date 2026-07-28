import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move up
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if grid[r, c] == 0:
                    continue
                target_r = r - 1
                if target_r >= 0 and grid[target_r, c] == 0:
                    new_grid[target_r, c] = grid[r, c]
                    new_grid[r, c] = 0
                elif target_r >= 0 and grid[target_r, c] != 0:
                    # Stack
                    new_grid[target_r, c] = grid[r, c]
                    new_grid[r, c] = 0
        return new_grid
    
    elif action == 2:
        # Move down
        for c in range(W):
            for r in range(H):
                if grid[r, c] == 0:
                    continue
                target_r = r + 1
                if target_r < H and grid[target_r, c] == 0:
                    new_grid[target_r, c] = grid[r, c]
                    new_grid[r, c] = 0
                elif target_r < H and grid[target_r, c] != 0:
                    # Stack
                    new_grid[target_r, c] = grid[r, c]
                    new_grid[r, c] = 0
        return new_grid
    
    elif action == 3:
        # Move left
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if grid[r, c] == 0:
                    continue
                target_c = c - 1
                if target_c >= 0 and grid[r, target_c] == 0:
                    new_grid[r, target_c] = grid[r, c]
                    new_grid[r, c] = 0
                elif target_c >= 0 and grid[r, target_c] != 0:
                    # Stack
                    new_grid[r, target_c] = grid[r, c]
                    new_grid[r, c] = 0
        return new_grid
    
    elif action == 4:
        # Move right
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 0:
                    continue
                target_c = c + 1
                if target_c < W and grid[r, target_c] == 0:
                    new_grid[r, target_c] = grid[r, c]
                    new_grid[r, c] = 0
                elif target_c < W and grid[r, target_c] != 0:
                    # Stack
                    new_grid[r, target_c] = grid[r, c]
                    new_grid[r, c] = 0
        return new_grid
    
    elif action == 5:
        # Move up-left
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if grid[r, c] == 0:
                    continue
                target_r = r - 1
                target_c = c - 1
                if target_r >= 0 and target_c >= 0 and grid[target_r, target_c] == 0:
                    new_grid[target_r, target_c] = grid[r, c]
                    new_grid[r, c] = 0
                elif target_r >= 0 and target_c >= 0 and grid[target_r, target_c] != 0:
                    # Stack
                    new_grid[target_r, target_c] = grid[r, c]
                    new_grid[r, c] = 0
        return new_grid
    
    elif action == 6:
        # Click (data provided)
        if data:
            px, py = data['x'], data['y']
            # Convert pixel to logical
            r, c = py // 1, px // 1
            if 0 <= r < H and 0 <= c < W:
                new_grid[r, c] = 0
        return new_grid
    
    elif action == 7:
        # Move up-right
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if grid[r, c] == 0:
                    continue
                target_r = r - 1
                target_c = c + 1
                if target_r >= 0 and target_c < W and grid[target_r, target_c] == 0:
                    new_grid[target_r, target_c] = grid[r, c]
                    new_grid[r, c] = 0
                elif target_r >= 0 and target_c < W and grid[target_r, target_c] != 0:
                    # Stack
                    new_grid[target_r, target_c] = grid[r, c]
                    new_grid[r, c] = 0
        return new_grid
    
    return grid

def is_level_complete(grid):
    # Check if all non-zero cells are in the top rows (rows 0-15)
    # Based on the initial grid structure
    H, W = grid.shape
    for r in range(H):
        if r > 15:
            if np.any(grid[r, :] != 0):
                return False
    return True