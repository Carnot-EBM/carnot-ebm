import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move Up
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if new_grid[r, c] == 0:
                    for prev_r in range(r - 1, -1, -1):
                        if new_grid[prev_r, c] != 0:
                            new_grid[r, c] = new_grid[prev_r, c]
                            new_grid[prev_r, c] = 0
                            break
    elif action == 2:
        # Action 2: Move Down
        for c in range(W):
            for r in range(H):
                if new_grid[r, c] == 0:
                    for prev_r in range(r + 1, H):
                        if new_grid[prev_r, c] != 0:
                            new_grid[r, c] = new_grid[prev_r, c]
                            new_grid[prev_r, c] = 0
                            break
    elif action == 3:
        # Action 3: Move Left
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if new_grid[r, c] == 0:
                    for prev_c in range(c - 1, -1, -1):
                        if new_grid[r, prev_c] != 0:
                            new_grid[r, c] = new_grid[r, prev_c]
                            new_grid[r, prev_c] = 0
                            break
    elif action == 4:
        # Action 4: Move Right
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 0:
                    for prev_c in range(c + 1, W):
                        if new_grid[r, prev_c] != 0:
                            new_grid[r, c] = new_grid[r, prev_c]
                            new_grid[r, prev_c] = 0
                            break
    elif action == 5:
        # Action 5: Move Up-Left
        for r in range(H - 1, -1, -1):
            for c in range(W - 1, -1, -1):
                if new_grid[r, c] == 0:
                    for prev_r in range(r - 1, -1, -1):
                        for prev_c in range(c - 1, -1, -1):
                            if new_grid[prev_r, prev_c] != 0:
                                new_grid[r, c] = new_grid[prev_r, prev_c]
                                new_grid[prev_r, prev_c] = 0
                                break
                        if new_grid[r, c] != 0:
                            break
    elif action == 6:
        # Action 6: Click (no-op in this model)
        pass
    elif action == 7:
        # Action 7: Move Up-Right
        for r in range(H - 1, -1, -1):
            for c in range(W):
                if new_grid[r, c] == 0:
                    for prev_r in range(r - 1, -1, -1):
                        for prev_c in range(c + 1, W):
                            if new_grid[prev_r, prev_c] != 0:
                                new_grid[r, c] = new_grid[prev_r, prev_c]
                                new_grid[prev_r, prev_c] = 0
                                break
                        if new_grid[r, c] != 0:
                            break
    
    return new_grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on the observed transitions, the win state is when the grid is fully filled
    # or when specific patterns are completed.
    # Given the complexity, we assume the win state is when the grid is fully filled with non-zero values.
    return np.all(grid != 0)