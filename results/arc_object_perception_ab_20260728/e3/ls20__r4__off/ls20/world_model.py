import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if new_grid[r, c] == 0:
                    for pr in range(r - 1, -1, -1):
                        if new_grid[pr, c] != 0:
                            new_grid[r, c] = new_grid[pr, c]
                            new_grid[pr, c] = 0
                            break
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H):
                if new_grid[r, c] == 0:
                    for pr in range(r + 1, H):
                        if new_grid[pr, c] != 0:
                            new_grid[r, c] = new_grid[pr, c]
                            new_grid[pr, c] = 0
                            break
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if new_grid[r, c] == 0:
                    for pc in range(c - 1, -1, -1):
                        if new_grid[r, pc] != 0:
                            new_grid[r, c] = new_grid[r, pc]
                            new_grid[r, pc] = 0
                            break
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 0:
                    for pc in range(c + 1, W):
                        if new_grid[r, pc] != 0:
                            new_grid[r, c] = new_grid[r, pc]
                            new_grid[pc, c] = 0
                            break
    elif action == 5:
        # Toggle 0/1
        new_grid = 1 - new_grid
    elif action == 6:
        # Click
        px, py = data['x'], data['y']
        new_grid[py, px] = 1 - new_grid[py, px]
    elif action == 7:
        # Toggle 0/1 (same as 5)
        new_grid = 1 - new_grid
        
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    target = np.zeros((H, W), dtype=int)
    target[5, :] = 4
    target[6, :] = 4
    target[7, :] = 4
    target[8, :] = 4
    target[9, :] = 4
    target[10, :] = 4
    target[11, :] = 4
    target[12, :] = 4
    target[13, :] = 4
    target[14, :] = 4
    target[15, :] = 4
    target[16, :] = 4
    target[17, :] = 4
    target[18, :] = 4
    target[19, :] = 4
    target[20, :] = 4
    target[21, :] = 4
    target[22, :] = 4
    target[23, :] = 4
    target[24, :] = 4
    target[25, :] = 4
    target[26, :] = 4
    target[27, :] = 4
    target[28, :] = 4
    target[29, :] = 4
    target[30, :] = 4
    target[31, :] = 4
    target[32, :] = 4
    target[33, :] = 4
    target[34, :] = 4
    target[35, :] = 4
    target[36, :] = 4
    target[37, :] = 4
    target[38, :] = 4
    target[39, :] = 4
    target[40, :] = 4
    target[41, :] = 4
    target[42, :] = 4
    target[43, :] = 4
    target[44, :] = 4
    target[45, :] = 4
    target[46, :] = 4
    target[47, :] = 4
    target[48, :] = 4
    target[49, :] = 4
    target[50, :] = 4
    target[51, :] = 4
    target[52, :] = 4
    target[53, :] = 4
    target[54, :] = 4
    target[55, :] = 4
    target[56, :] = 4
    target[57, :] = 4
    target[58, :] = 4
    target[59, :] = 4
    target[60, :] = 4
    target[61, :] = 4
    target[62, :] = 4
    target[63, :] = 4
    
    return np.array_equal(grid, target)