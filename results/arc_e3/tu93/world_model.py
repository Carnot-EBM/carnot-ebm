import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Collect 14
        if grid[63, 63] == 14:
            new_grid[63, 63] = 0
            new_grid[63, 62] = 14
            new_grid[63, 61] = 14
            new_grid[63, 60] = 14
            new_grid[63, 59] = 14
            new_grid[63, 58] = 14
            new_grid[63, 57] = 14
            new_grid[63, 56] = 14
    elif action == 2:
        # Move Left
        new_grid = move_direction(new_grid, 0)
    elif action == 3:
        # Move Right
        new_grid = move_direction(new_grid, 1)
    elif action == 4:
        # Move Up
        new_grid = move_direction(new_grid, -1)
    elif action == 5:
        # Move Down
        new_grid = move_direction(new_grid, 1)
    elif action == 6:
        # Click
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            new_grid[py, px] = 14
    elif action == 7:
        # Move Diagonal (Down-Right)
        new_grid = move_direction(new_grid, 1)
        new_grid = move_direction(new_grid, 1)
    
    return new_grid

def move_direction(grid, direction):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if direction == 0:
        # Left
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 5:
                    new_grid[r, c] = 0
                    new_grid[r, c-1] = grid[r, c]
                    new_grid[r, c] = 0
    elif direction == 1:
        # Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 5:
                    new_grid[r, c] = 0
                    new_grid[r, c+1] = grid[r, c]
                    new_grid[r, c] = 0
    elif direction == -1:
        # Up
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 5:
                    new_grid[r, c] = 0
                    new_grid[r-1, c] = grid[r, c]
                    new_grid[r, c] = 0
    elif direction == 1:
        # Down
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 5:
                    new_grid[r, c] = 0
                    new_grid[r+1, c] = grid[r, c]
                    new_grid[r, c] = 0
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all cells are 5
    return np.all(grid == 5)