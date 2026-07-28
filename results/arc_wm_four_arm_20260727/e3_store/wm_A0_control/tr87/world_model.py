import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        if 0 <= logical_y < H and 0 <= logical_x < W:
            grid[logical_y, logical_x] = 7
    elif action == 2:
        grid = grid.copy()
        for _ in range(10):
            moved = False
            for r in range(H - 1, -1, -1):
                for c in range(W - 1, -1, -1):
                    if grid[r, c] == 7:
                        next_r, next_c = r + 1, c
                        if next_r < H and grid[next_r, next_c] != 7:
                            grid[r, c] = 0
                            grid[next_r, next_c] = 7
                            moved = True
            if not moved:
                break
    elif action == 3:
        grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 7:
                    grid[r, c] = 0
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    count = np.sum(grid == 7)
    return count >= 10