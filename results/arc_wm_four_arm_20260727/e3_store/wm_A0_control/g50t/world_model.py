import numpy as np

def engine(grid, action, data):
    if action == 1:
        return apply_action_1(grid)
    elif action == 2:
        return apply_action_2(grid)
    elif action == 3:
        return apply_action_3(grid)
    elif action == 4:
        return apply_action_4(grid)
    elif action == 6:
        return apply_action_6(grid, data)
    else:
        return grid

def apply_action_1(grid):
    h, w = grid.shape
    new_grid = grid.copy()
    # Action 1: Toggle cell at (63, 62)
    if 63 < h and 62 < w:
        new_grid[63, 62] = 12 if new_grid[63, 62] != 12 else 0
    return new_grid

def apply_action_2(grid):
    h, w = grid.shape
    new_grid = grid.copy()
    # Action 2: Toggle cells in a specific pattern
    # Based on observed changes: r8c14, r9c14, r10c14, r10c17, r11c14, r12c14, r14c14, r15c14, r16c14, r16c17, r17c14, r18c14
    # Pattern: Toggle cells in a vertical line at col 14, and some at col 17
    # Rows affected: 8, 9, 10, 11, 12, 14, 15, 16, 17, 18
    rows = [8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
    cols = [14, 17]
    for r in rows:
        for c in cols:
            if 0 <= r < h and 0 <= c < w:
                new_grid[r, c] = 15 if new_grid[r, c] != 15 else 0
    return new_grid

def apply_action_3(grid):
    h, w = grid.shape
    new_grid = grid.copy()
    # Action 3: Toggle cells in a specific pattern
    # Based on observed changes: r8c14, r8c20, r9c14, r9c20, r10c14, r10c17, r10c20, r10c23, r11c14, r11c20, r12c14, r12c20
    # Pattern: Toggle cells in a grid-like pattern
    rows = [8, 9, 10, 11, 12]
    cols = [14, 17, 20, 23]
    for r in rows:
        for c in cols:
            if 0 <= r < h and 0 <= c < w:
                new_grid[r, c] = 5 if new_grid[r, c] != 5 else 0
    return new_grid

def apply_action_4(grid):
    h, w = grid.shape
    new_grid = grid.copy()
    # Action 4: Toggle cells in a specific pattern
    # Based on observed changes: r8c14, r8c20, r9c14, r9c20, r10c14, r10c20, r10c23, r11c14, r11c20, r12c14, r12c20, r63c63
    # Pattern: Toggle cells in a grid-like pattern plus one at (63, 63)
    rows = [8, 9, 10, 11, 12]
    cols = [14, 20, 23]
    for r in rows:
        for c in cols:
            if 0 <= r < h and 0 <= c < w:
                new_grid[r, c] = 5 if new_grid[r, c] != 5 else 0
    new_grid[63, 63] = 12 if new_grid[63, 63] != 12 else 0
    return new_grid

def apply_action_6(grid, data):
    h, w = grid.shape
    new_grid = grid.copy()
    # Action 6: Click at pixel coordinates (px, py)
    if data and 'x' in data and 'y' in data:
        px, py = data['x'], data['y']
        if 0 <= py < h and 0 <= px < w:
            new_grid[py, px] = 12 if new_grid[py, px] != 12 else 0
    return new_grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on the initial grid, the win state is when all cells are color 15
    return np.all(grid == 15)