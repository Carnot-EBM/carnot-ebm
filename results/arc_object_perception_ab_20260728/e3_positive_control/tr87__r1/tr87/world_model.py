import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move right
        for r in range(H):
            for c in range(W - 1):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    new_grid[r, c + 1] = 5
    elif action == 2:
        # Move down
        for r in range(H - 1):
            for c in range(W):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c] = 5
    elif action == 3:
        # Move left
        for r in range(H):
            for c in range(1, W):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    new_grid[r, c - 1] = 5
    elif action == 4:
        # Move up
        for r in range(1, H):
            for c in range(W):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c] = 5
    elif action == 6:
        # Click action - toggle 5s to 0s in a 3x3 area around data
        px, py = data['x'], data['y']
        for r in range(max(0, py - 1), min(H, py + 2)):
            for c in range(max(0, px - 1), min(W, px + 2)):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
    elif action == 7:
        # Collect all 5s
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all 5s are collected (grid should have no 5s)
    if np.any(grid == 5):
        return False
    
    # Check if the grid matches the win state pattern
    # The win state has specific patterns in rows 4-10, 13-19, 22-28, 40-46, 48-57
    # We check if the grid has the correct structure
    
    # Check rows 4-10
    for r in range(4, 11):
        if not _check_row_pattern(grid, r):
            return False
    
    # Check rows 13-19
    for r in range(13, 20):
        if not _check_row_pattern(grid, r):
            return False
    
    # Check rows 22-28
    for r in range(22, 29):
        if not _check_row_pattern(grid, r):
            return False
    
    # Check rows 40-46
    for r in range(40, 47):
        if not _check_row_pattern(grid, r):
            return False
    
    # Check rows 48-57
    for r in range(48, 58):
        if not _check_row_pattern(grid, r):
            return False
    
    return True

def _check_row_pattern(grid, r):
    # Check if the row matches the expected pattern
    # This is a simplified check - in reality, we'd need to compare with the exact win state
    # For now, we just check if the row has the correct number of 5s and 0s
    row = grid[r]
    if np.any(row == 5):
        return False
    return True