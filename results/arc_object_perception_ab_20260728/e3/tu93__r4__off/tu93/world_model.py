import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 2:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Action 2: Toggle cell at (py, px)
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 14
    elif action == 3:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Action 3: Toggle cell at (py, px)
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 14
    elif action == 4:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Action 4: Toggle cell at (py, px)
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 14
    elif action == 5:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Action 5: Toggle cell at (py, px)
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 14
    elif action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Action 6: Toggle cell at (py, px)
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 14
    elif action == 7:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Action 7: Toggle cell at (py, px)
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 14
    elif action == 1:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Action 1: Toggle cell at (py, px)
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 14
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all cells are 5 or 14
    if np.all((grid == 5) | (grid == 14)):
        return True
    return False