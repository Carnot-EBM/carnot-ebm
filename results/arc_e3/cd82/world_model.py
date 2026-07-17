import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] == 5:
                    new_grid[r, c] = grid[r - 1, c]
                    new_grid[r - 1, c] = 5
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H - 1):
                if grid[r, c] == 5:
                    new_grid[r, c] = grid[r + 1, c]
                    new_grid[r + 1, c] = 5
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if grid[r, c] == 5:
                    new_grid[r, c] = grid[r, c - 1]
                    new_grid[r, c - 1] = 5
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    new_grid[r, c] = grid[r, c + 1]
                    new_grid[r, c + 1] = 5
    elif action == 5:
        # Move Up-Left
        for r in range(H - 1, -1, -1):
            for c in range(W - 1, 0, -1):
                if grid[r, c] == 5:
                    new_grid[r, c] = grid[r - 1, c - 1]
                    new_grid[r - 1, c - 1] = 5
    elif action == 6:
        # Click
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            new_grid[py, px] = 5
    elif action == 7:
        # Move Up-Right
        for r in range(H - 1, -1, -1):
            for c in range(W):
                if grid[r, c] == 5:
                    new_grid[r, c] = grid[r - 1, c + 1]
                    new_grid[r - 1, c + 1] = 5
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        row_str = ""
        col = 0
        if r < 24:
            if r < 3:
                row_str = f"r{r}:5x16,4x2,3x{W-18}"
            elif r < 7:
                row_str = f"r{r}:5x3,0x10,5x3,4x2,3x{W-18},4x1,0x3,4x1,3x1,4x1,15x3,4x1,3x18"
            elif r < 13:
                row_str = f"r{r}:5x3,15x10,5x3,4x2,5x{W-18}"
            elif r < 16:
                row_str = f"r{r}:5x16,4x2,5x{W-18}"
            elif r < 18:
                row_str = f"r{r}:5x{W}"
            elif r < 24:
                row_str = f"r{r}:5x25,2x14,5x25"
            elif r < 33:
                row_str = f"r{r}:5x25,2x1,15x12,2x1,5x25"
            elif r < 34:
                row_str = f"r{r}:5x27,0x10,5x27"
            elif r < 44:
                row_str = f"r{r}:5x{W}"
            elif r < 63:
                row_str = f"r{r}:5x{W}"
            else:
                row_str = f"r{r}:4x{W}"
        else:
            row_str = f"r{r}:5x{W}"
        
        if row_str != f"r{r}:{grid[r].tolist()}":
            return False
    return True