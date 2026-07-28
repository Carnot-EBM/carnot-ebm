import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        if grid[py, px] == 4:
            new_grid = grid.copy()
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    if 0 <= py + dy < H and 0 <= px + dx < W:
                        if grid[py + dy, px + dx] == 4:
                            new_grid[py + dy, px + dx] = 0
            return new_grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for r in range(64):
        row = grid[r, :]
        if row[0] != 4:
            return False
        if row[63] != 4:
            return False
        for c in range(1, 63):
            if row[c] != 4 and row[c] != 5:
                return False
    for r in range(8, 16):
        if not np.all(grid[r, :] == 4):
            return False
    for r in range(36, 64):
        if not np.all(grid[r, :] == 4):
            return False
    return True