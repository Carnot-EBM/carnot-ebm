import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move right
        if H > 0 and W > 0:
            new_grid[0, -1] = 5
    elif action == 2:
        # Action 2: Move left
        if H > 0 and W > 0:
            new_grid[0, 0] = 5
    elif action == 3:
        # Action 3: Move down
        if H > 0 and W > 0:
            new_grid[-1, -1] = 5
    elif action == 4:
        # Action 4: Move up
        if H > 0 and W > 0:
            new_grid[0, 0] = 5
    elif action == 5:
        # Action 5: Move right-down
        if H > 0 and W > 0:
            new_grid[-1, -1] = 5
    elif action == 6:
        # Action 6: Click (data provided)
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            if 0 <= px < W and 0 <= py < H:
                new_grid[py, px] = 5
    elif action == 7:
        # Action 7: Move left-up
        if H > 0 and W > 0:
            new_grid[0, 0] = 5
            
    return new_grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # Based on the observed transitions, the win state involves filling specific cells with 5
    # The win state is reached when the grid has a specific configuration
    # Since we don't have the exact win state grid, we assume it's when the grid is fully filled with 5s
    # or a specific pattern. However, based on the transitions, it seems like the game is about filling cells with 5s.
    # Let's assume the win state is when the grid is fully filled with 5s.
    return np.all(grid == 5)