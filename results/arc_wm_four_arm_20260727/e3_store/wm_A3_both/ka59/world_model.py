import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 0
    elif action == 1:
        # Move up
        for r in range(H - 1, -1, -1):
            for c in range(W):
                if new_grid[r, c] == 10:
                    for nr in range(r - 1, -1, -1):
                        if new_grid[nr, c] == 15:
                            new_grid[nr, c] = 10
                            new_grid[r, c] = 15
                            break
    elif action == 4:
        # Move down
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 10:
                    for nr in range(r + 1, H):
                        if new_grid[nr, c] == 15:
                            new_grid[nr, c] = 10
                            new_grid[r, c] = 15
                            break
    elif action == 2:
        # Move left
        for c in range(W - 1, -1, -1):
            for r in range(H):
                if new_grid[r, c] == 10:
                    for nc in range(c - 1, -1, -1):
                        if new_grid[r, nc] == 15:
                            new_grid[r, nc] = 10
                            new_grid[r, c] = 15
                            break
    elif action == 3:
        # Move right
        for c in range(W):
            for r in range(H):
                if new_grid[r, c] == 10:
                    for nc in range(c + 1, W):
                        if new_grid[r, nc] == 15:
                            new_grid[r, nc] = 10
                            new_grid[r, c] = 15
                            break
    elif action == 5:
        # Toggle 10 <-> 15
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 10:
                    new_grid[r, c] = 15
                elif new_grid[r, c] == 15:
                    new_grid[r, c] = 10
    elif action == 7:
        # Collect 10s (turn to 0)
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 10:
                    new_grid[r, c] = 0
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    count_10 = 0
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 10:
                count_10 += 1
    return count_10 == 0