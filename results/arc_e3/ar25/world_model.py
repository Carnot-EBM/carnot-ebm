import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 2:
        # Action 2: Move right, push blocks
        for r in range(H):
            for c in range(W - 1):
                if grid[r, c] != 0 and grid[r, c + 1] == 0:
                    new_grid[r, c + 1] = grid[r, c]
                    new_grid[r, c] = 0
                elif grid[r, c] != 0 and grid[r, c + 1] != 0 and grid[r, c + 1] == grid[r, c]:
                    new_grid[r, c + 1] = grid[r, c]
                    new_grid[r, c] = 0
    elif action == 3:
        # Action 3: Move left, push blocks
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if grid[r, c] != 0 and grid[r, c - 1] == 0:
                    new_grid[r, c - 1] = grid[r, c]
                    new_grid[r, c] = 0
                elif grid[r, c] != 0 and grid[r, c - 1] != 0 and grid[r, c - 1] == grid[r, c]:
                    new_grid[r, c - 1] = grid[r, c]
                    new_grid[r, c] = 0
    elif action == 4:
        # Action 4: Move up, push blocks
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] != 0 and grid[r - 1, c] == 0:
                    new_grid[r - 1, c] = grid[r, c]
                    new_grid[r, c] = 0
                elif grid[r, c] != 0 and grid[r - 1, c] != 0 and grid[r - 1, c] == grid[r, c]:
                    new_grid[r - 1, c] = grid[r, c]
                    new_grid[r, c] = 0
    elif action == 7:
        # Action 7: Move down, push blocks
        for c in range(W):
            for r in range(H - 1):
                if grid[r, c] != 0 and grid[r + 1, c] == 0:
                    new_grid[r + 1, c] = grid[r, c]
                    new_grid[r, c] = 0
                elif grid[r, c] != 0 and grid[r + 1, c] != 0 and grid[r + 1, c] == grid[r, c]:
                    new_grid[r + 1, c] = grid[r, c]
                    new_grid[r, c] = 0
    elif action == 6:
        # Action 6: Click at pixel coordinates (x, y)
        px, py = data['x'], data['y']
        new_grid[py, px] = 0
    # Actions 1 and 5 are not observed but follow similar logic if needed
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if the grid is full of color 9 (the target color)
    return np.all(grid == 9)