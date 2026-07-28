import numpy as np

import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move Up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] != 0:
                    if grid[r - 1, c] == 0:
                        new_grid[r, c] = 0
                        new_grid[r - 1, c] = grid[r, c]
                        grid[r, c] = 0
                        grid[r - 1, c] = grid[r - 1, c]
        return new_grid
    elif action == 2:
        # Action 2: Move Down
        for c in range(W):
            for r in range(H - 1):
                if grid[r, c] != 0:
                    if grid[r + 1, c] == 0:
                        new_grid[r, c] = 0
                        new_grid[r + 1, c] = grid[r, c]
                        grid[r, c] = 0
                        grid[r + 1, c] = grid[r + 1, c]
        return new_grid
    elif action == 3:
        # Action 3: Move Left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if grid[r, c] != 0:
                    if grid[r, c - 1] == 0:
                        new_grid[r, c] = 0
                        new_grid[r, c - 1] = grid[r, c]
                        grid[r, c] = 0
                        grid[r, c - 1] = grid[r, c - 1]
        return new_grid
    elif action == 4:
        # Action 4: Move Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 0:
                    if grid[r, c + 1] == 0:
                        new_grid[r, c] = 0
                        new_grid[r, c + 1] = grid[r, c]
                        grid[r, c] = 0
                        grid[r, c + 1] = grid[r, c + 1]
        return new_grid
    elif action == 5:
        # Action 5: Toggle 0 <-> 10
        mask = (grid == 0) | (grid == 10)
        new_grid[mask] = 10 if grid[mask] == 0 else 0
        return new_grid
    elif action == 6:
        # Action 6: Click (data={'x': px, 'y': py})
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            if 0 <= py < H and 0 <= px < W:
                new_grid[py, px] = 10
        return new_grid
    elif action == 7:
        # Action 7: Toggle 0 <-> 11
        mask = (grid == 0) | (grid == 11)
        new_grid[mask] = 11 if grid[mask] == 0 else 0
        return new_grid
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if the grid matches the win state pattern
    # Win state has specific structure:
    # - Rows 0-14: 9x36, 10x3, 9x24, 11x1
    # - Rows 15-17: 9x36, 10x3, 9x24, 11x1
    # - Rows 18-23: 9x21, 4x9, 9x6, 10x3, 9x6, 5x9, 9x9, 11x1
    # - Rows 24-26: 9x21, 4x3, 9x12, 10x3, 9x12, 5x3, 9x9, 11x1
    # - Rows 27-29: 9x15, 4x9, 9x12, 10x3, 9x12, 5x9, 9x3, 11x1
    # - Rows 30-39: 9x36, 10x3, 9x24, 11x1
    # - Rows 40-41: 9x36, 10x3, 9x24, 11x1
    # - Rows 42-44: 9x9, 11x9, 9x18, 10x3, 9x24, 11x1
    # - Rows 45-47: 9x9, 11x3, 9x24, 10x3, 9x24, 11x1
    # - Rows 48-50: 9x9, 11x3, 9x24, 10x3, 9x24, 11x1
    # - Rows 51-53: 9x3, 11x9, 9x24, 10x3, 9x24, 11x1
    # - Rows 54-62: 9x36, 10x3, 9x24, 11x1
    
    # Check row patterns
    for r in range(H):
        row = grid[r, :]
        if r in range(0, 18):
            # Pattern: 9x36, 10x3, 9x24, 11x1
            if not (np.all(row[:36] == 9) and 
                    np.all(row[36:39] == 10) and 
                    np.all(row[39:63] == 9) and 
                    row[63] == 11):
                return False
        elif r in range(18, 24):
            # Pattern: 9x21, 4x9, 9x6, 10x3, 9x6, 5x9, 9x9, 11x1
            if not (np.all(row[:21] == 9) and 
                    np.all(row[21:30] == 4) and 
                    np.all(row[30:36] == 9) and 
                    np.all(row[36:39] == 10) and 
                    np.all(row[39:45] == 9) and 
                    np.all(row[45:54] == 5) and 
                    np.all(row[54:63] == 9) and 
                    row[63] == 11):
                return False
        elif r in range(24, 30):
            # Pattern: 9x21, 4x3, 9x12, 10x3, 9x12, 5x3, 9x9, 11x1
            if not (np.all(row[:21] == 9) and 
                    np.all(row[21:24] == 4) and 
                    np.all(row[24:36] == 9) and 
                    np.all(row[36:39] == 10) and 
                    np.all(row[39:51] == 9) and 
                    np.all(row[51:54] == 5) and 
                    np.all(row[54:63] == 9) and 
                    row[63] == 11):
                return False
        elif r in range(30, 40):
            # Pattern: 9x36, 10x3, 9x24, 11x1
            if not (np.all(row[:36] == 9) and 
                    np.all(row[36:39] == 10) and 
                    np.all(row[39:63] == 9) and 
                    row[63] == 11):
                return False
        elif r in range(40, 42):
            # Pattern: 9x36, 10x3, 9x24, 11x1
            if not (np.all(row[:36] == 9) and 
                    np.all(row[36:39] == 10) and 
                    np.all(row[39:63] == 9) and 
                    row[63] == 11):
                return False
        elif r in range(42, 45):
            # Pattern: 9x9, 11x9, 9x18, 10x3, 9x24, 11x1
            if not (np.all(row[:9] == 9) and 
                    np.all(row[9:18] == 11) and 
                    np.all(row[18:36] == 9) and 
                    np.all(row[36:39] == 10) and 
                    np.all(row[39:63] == 9) and 
                    row[63] == 11):
                return False
        elif r in range(45, 48):
            # Pattern: 9x9, 11x3, 9x24, 10x3, 9x24, 11x1
            if not (np.all(row[:9] == 9) and 
                    np.all(row[9:12] == 11) and 
                    np.all(row[12:36] == 9) and 
                    np.all(row[36:39] == 10) and 
                    np.all(row[39:63] == 9) and 
                    row[63] == 11):
                return False
        elif r in range(48, 51):
            # Pattern: 9x9, 11x3, 9x24, 10x3, 9x24, 11x1
            if not (np.all(row[:9] == 9) and 
                    np.all(row[9:12] == 11) and 
                    np.all(row[12:36] == 9) and 
                    np.all(row[36:39] == 10) and 
                    np.all(row[39:63] == 9) and 
                    row[63] == 11):
                return False
        elif r in range(51, 54):
            # Pattern: 9x3, 11x9, 9x24, 10x3, 9x24, 11x1
            if not (np.all(row[:3] == 9) and 
                    np.all(row[3:12] == 11) and 
                    np.all(row[12:36] == 9) and 
                    np.all(row[36:39] == 10) and 
                    np.all(row[39:63] == 9) and 
                    row[63] == 11):
                return False
        elif r in range(54, 63):
            # Pattern: 9x36, 10x3, 9x24, 11x1
            if not (np.all(row[:36] == 9) and 
                    np.all(row[36:39] == 10) and 
                    np.all(row[39:63] == 9) and 
                    row[63] == 11):
                return False
    
    # Check last row
    if not (np.all(grid[63, :] == 5) and grid[63, 63] == 11):
        return False
    
    return True

def is_level_complete(grid):
    import numpy as np
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
    if grid.shape != (10, 10):
        return False
    if not np.all(grid == 0):
        return False
    return True
