import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if new_grid[r, c] == 0 and new_grid[r - 1, c] != 0:
                    new_grid[r, c] = new_grid[r - 1, c]
                    new_grid[r - 1, c] = 0
        return new_grid
    
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H - 1):
                if new_grid[r, c] == 0 and new_grid[r + 1, c] != 0:
                    new_grid[r, c] = new_grid[r + 1, c]
                    new_grid[r + 1, c] = 0
        return new_grid
    
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if new_grid[r, c] == 0 and new_grid[r, c - 1] != 0:
                    new_grid[r, c] = new_grid[r, c - 1]
                    new_grid[r, c - 1] = 0
        return new_grid
    
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 0 and new_grid[r, c + 1] != 0:
                    new_grid[r, c] = new_grid[r, c + 1]
                    new_grid[r, c + 1] = 0
        return new_grid
    
    elif action == 6:
        # Click (data is {'x': px, 'y': py})
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            # Assuming click toggles or affects specific cell
            # Based on observations, clicks seem to trigger specific changes
            # For simplicity, we assume it toggles the cell at (py, px)
            if 0 <= py < H and 0 <= px < W:
                if new_grid[py, px] == 0:
                    new_grid[py, px] = 15
                else:
                    new_grid[py, px] = 0
        return new_grid
    
    elif action == 7:
        # Action 7 is not explicitly defined in the observations but we can assume it's similar to others
        return new_grid
    
    return new_grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on observations, the win state seems to be when the grid is fully filled with a specific color
    # or when certain conditions are met.
    # From the initial grid, row 63 is all 15s.
    # Let's assume the level is complete if the last row is all 15s.
    return np.all(grid[-1, :] == 15)