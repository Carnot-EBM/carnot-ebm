import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if grid[r, c] == 10:
                    for nr in range(r - 1, -1, -1):
                        if grid[nr, c] != 10:
                            new_grid[r, c] = grid[nr, c]
                            new_grid[nr, c] = 10
                            break
                    else:
                        new_grid[r, c] = 10
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if grid[r, c] == 10:
                    for nc in range(c - 1, -1, -1):
                        if grid[r, nc] != 10:
                            new_grid[r, c] = grid[r, nc]
                            new_grid[r, nc] = 10
                            break
                    else:
                        new_grid[r, c] = 10
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 10:
                    for nc in range(c + 1, W):
                        if grid[r, nc] != 10:
                            new_grid[r, c] = grid[r, nc]
                            new_grid[r, nc] = 10
                            break
                    else:
                        new_grid[r, c] = 10
    elif action == 7:
        # Move Down
        for c in range(W):
            for r in range(H):
                if grid[r, c] == 10:
                    for nr in range(r + 1, H):
                        if grid[nr, c] != 10:
                            new_grid[r, c] = grid[nr, c]
                            new_grid[nr, c] = 10
                            break
                    else:
                        new_grid[r, c] = 10
    elif action == 6:
        # Click
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 5
    elif action == 2:
        # Toggle
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 10 if new_grid[py, px] == 5 else 5
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 10:
                return False
    return True