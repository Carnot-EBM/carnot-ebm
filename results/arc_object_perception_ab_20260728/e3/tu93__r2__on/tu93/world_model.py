import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 2:
        # Action 2: Move player left (x-1)
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if px > 0:
            new_grid[py, px - 1] = 6
            new_grid[py, px] = 5
    elif action == 3:
        # Action 3: Move player up (y-1)
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if py > 0:
            new_grid[py - 1, px] = 6
            new_grid[py, px] = 5
    elif action == 4:
        # Action 4: Move player right (x+1)
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if px < W - 1:
            new_grid[py, px + 1] = 6
            new_grid[py, px] = 5
    elif action == 5:
        # Action 5: Move player down (y+1)
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if py < H - 1:
            new_grid[py + 1, px] = 6
            new_grid[py, px] = 5
    elif action == 6:
        # Action 6: Click to collect
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            if new_grid[py, px] != 5:
                new_grid[py, px] = 5
    elif action in [1, 7]:
        # Actions 1 and 7: No-op or special (no change)
        pass
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all cells are filled with 5 or 6
    if np.any((grid != 5) & (grid != 6)):
        return False
    # Check if the player (6) is at the bottom-right corner
    if grid[H - 1, W - 1] != 6:
        return False
    return True