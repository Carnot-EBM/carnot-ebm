import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 1:
        # Move Up
        changed = np.zeros_like(grid)
        for r in range(H - 1, -1, -1):
            for c in range(W):
                if grid[r, c] == 0:
                    above = r - 1
                    if above >= 0 and grid[above, c] != 0:
                        changed[r, c] = grid[above, c]
                        changed[above, c] = 0
        return grid + changed
    elif action == 2:
        # Move Down
        changed = np.zeros_like(grid)
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 0:
                    below = r + 1
                    if below < H and grid[below, c] != 0:
                        changed[r, c] = grid[below, c]
                        changed[below, c] = 0
        return grid + changed
    elif action == 3:
        # Move Left
        changed = np.zeros_like(grid)
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if grid[r, c] == 0:
                    left = c - 1
                    if left >= 0 and grid[r, left] != 0:
                        changed[r, c] = grid[r, left]
                        changed[r, left] = 0
        return grid + changed
    elif action == 4:
        # Move Right
        changed = np.zeros_like(grid)
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 0:
                    right = c + 1
                    if right < W and grid[r, right] != 0:
                        changed[r, c] = grid[r, right]
                        changed[r, right] = 0
        return grid + changed
    elif action == 6:
        # Click
        if data is not None:
            px, py = data['x'], data['y']
            if 0 <= py < H and 0 <= px < W:
                grid[py, px] = 0
        return grid
    return grid

def is_level_complete(grid):
    return False