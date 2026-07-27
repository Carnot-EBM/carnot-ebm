import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if grid[r, c] != 2:
                    for nr in range(r - 1, -1, -1):
                        if grid[nr, c] == 2:
                            new_grid[nr, c] = grid[r, c]
                            new_grid[r, c] = 2
                            break
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H):
                if grid[r, c] != 2:
                    for nr in range(r + 1, H):
                        if grid[nr, c] == 2:
                            new_grid[nr, c] = grid[r, c]
                            new_grid[r, c] = 2
                            break
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if grid[r, c] != 2:
                    for nc in range(c - 1, -1, -1):
                        if grid[r, nc] == 2:
                            new_grid[r, nc] = grid[r, c]
                            new_grid[r, c] = 2
                            break
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 2:
                    for nc in range(c + 1, W):
                        if grid[r, nc] == 2:
                            new_grid[r, nc] = grid[r, c]
                            new_grid[r, c] = 2
                            break
    elif action == 5:
        # Toggle
        if data is not None:
            px, py = data['x'], data['y']
            if 0 <= py < H and 0 <= px < W:
                new_grid[py, px] = 2 if new_grid[py, px] != 2 else 0
    elif action == 6:
        # Click
        if data is not None:
            px, py = data['x'], data['y']
            if 0 <= py < H and 0 <= px < W:
                new_grid[py, px] = 2 if new_grid[py, px] != 2 else 0
    elif action == 7:
        # Rotate
        new_grid = np.rot90(grid, k=1, axes=(0, 1))
    
    return new_grid

def is_level_complete(grid):
    return np.all(grid == 2)