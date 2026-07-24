import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 2:
        if data is None:
            return new_grid
        # Action 2: Move Down
        for r in range(H - 1, -1, -1):
            for c in range(W):
                if new_grid[r, c] == 5:
                    if r + 1 < H and new_grid[r + 1, c] != 5:
                        new_grid[r, c] = 0
                        new_grid[r + 1, c] = 5
    elif action == 3:
        if data is None:
            return new_grid
        # Action 3: Move Left
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if new_grid[r, c] == 5:
                    if c - 1 >= 0 and new_grid[r, c - 1] != 5:
                        new_grid[r, c] = 0
                        new_grid[r, c - 1] = 5
    elif action == 4:
        if data is None:
            return new_grid
        # Action 4: Move Right
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 5:
                    if c + 1 < W and new_grid[r, c + 1] != 5:
                        new_grid[r, c] = 0
                        new_grid[r, c + 1] = 5
    elif action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            if new_grid[py, px] == 5:
                new_grid[py, px] = 0
            else:
                new_grid[py, px] = 5
    elif action in [1, 5, 7]:
        # Actions 1, 5, 7: No-op or handled similarly to 2, 3, 4
        pass
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 5:
                return False
    return True