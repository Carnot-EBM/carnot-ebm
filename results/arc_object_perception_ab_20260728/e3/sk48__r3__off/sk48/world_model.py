import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move down (gravity)
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c] = 5
    elif action == 3:
        # Action 3: Move right
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c] = 5
    elif action == 4:
        # Action 4: Move left
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c] = 5
    elif action == 6:
        # Action 6: Click (toggle)
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            if 0 <= py < H and 0 <= px < W:
                if new_grid[py, px] == 5:
                    new_grid[py, px] = 0
                else:
                    new_grid[py, px] = 5
    elif action == 2:
        # Action 2: Move up
        for c in range(W):
            for r in range(H):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c] = 5
    elif action == 5:
        # Action 5: Move down-left
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c - 1] = 5
    elif action == 7:
        # Action 7: Move down-right
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c + 1] = 5
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    target_rows = 6
    target_cols = 42
    
    for r in range(H):
        if r < target_rows:
            if not (grid[r, :target_cols].sum() == 0 and grid[r, target_cols:].sum() == 5 * (W - target_cols)):
                return False
        elif r >= target_rows:
            if not (grid[r, :target_cols].sum() == 5 * target_cols and grid[r, target_cols:].sum() == 5 * (W - target_cols)):
                return False
    
    return True