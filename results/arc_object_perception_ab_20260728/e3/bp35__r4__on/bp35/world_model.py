import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 15
    elif action == 3:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 15
            # Apply gravity for color 10
            for r in range(H - 1, -1, -1):
                for c in range(W):
                    if new_grid[r, c] == 10:
                        # Find next empty or 10 spot below
                        found = False
                        for dr in range(r + 1, H):
                            if new_grid[dr, c] == 0:
                                new_grid[dr, c] = 10
                                new_grid[r, c] = 0
                                found = True
                                break
                            elif new_grid[dr, c] == 10:
                                continue
                        if not found:
                            new_grid[r, c] = 0
    else:
        return new_grid
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    
    # Check if row 63 is all 0
    if not np.all(grid[63, :] == 0):
        return False
    
    # Check if row 0 has the pattern 5x37, 10x17, 5x10
    row0 = grid[0, :]
    if not (np.sum(row0 == 5) == 37 and np.sum(row0 == 10) == 17 and np.sum(row0 == 5) == 10):
        return False
    
    # Check if row 62 has the pattern 5x2, 3x1, 5x4, 10x41, 5x2, 3x1, 5x5, 3x1, 5x1
    row62 = grid[62, :]
    if not (np.sum(row62 == 5) == 2 and np.sum(row62 == 3) == 1 and np.sum(row62 == 5) == 4 and np.sum(row62 == 10) == 41 and np.sum(row62 == 5) == 2 and np.sum(row62 == 3) == 1 and np.sum(row62 == 5) == 5 and np.sum(row62 == 3) == 1 and np.sum(row62 == 5) == 1):
        return False
    
    return True