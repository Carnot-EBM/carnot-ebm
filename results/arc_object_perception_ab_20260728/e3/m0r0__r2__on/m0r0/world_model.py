import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    
    if action == 1:
        # Action 1: Move player (color 0) one step UP (row - 1)
        # Find player
        player_pos = np.argwhere(grid == 0)
        if len(player_pos) == 0:
            return grid
        py, px = player_pos[0]
        if py > 0:
            new_grid = grid.copy()
            new_grid[py, px] = 5  # Player becomes color 5
            new_grid[py - 1, px] = 0  # New player position
            return new_grid
        return grid

    elif action == 2:
        # Action 2: Move player one step DOWN (row + 1)
        player_pos = np.argwhere(grid == 0)
        if len(player_pos) == 0:
            return grid
        py, px = player_pos[0]
        if py < H - 1:
            new_grid = grid.copy()
            new_grid[py, px] = 5
            new_grid[py + 1, px] = 0
            return new_grid
        return grid

    elif action == 3:
        # Action 3: Move player one step LEFT (col - 1)
        player_pos = np.argwhere(grid == 0)
        if len(player_pos) == 0:
            return grid
        py, px = player_pos[0]
        if px > 0:
            new_grid = grid.copy()
            new_grid[py, px] = 5
            new_grid[py, px - 1] = 0
            return new_grid
        return grid

    elif action == 4:
        # Action 4: Move player one step RIGHT (col + 1)
        player_pos = np.argwhere(grid == 0)
        if len(player_pos) == 0:
            return grid
        py, px = player_pos[0]
        if px < W - 1:
            new_grid = grid.copy()
            new_grid[py, px] = 5
            new_grid[py, px + 1] = 0
            return new_grid
        return grid

    elif action == 5:
        # Action 5: Move player one step UP-LEFT (diagonal)
        player_pos = np.argwhere(grid == 0)
        if len(player_pos) == 0:
            return grid
        py, px = player_pos[0]
        if py > 0 and px > 0:
            new_grid = grid.copy()
            new_grid[py, px] = 5
            new_grid[py - 1, px - 1] = 0
            return new_grid
        return grid

    elif action == 6:
        # Action 6: Click action - place color 5 at clicked position
        if 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            if 0 <= py < H and 0 <= px < W:
                new_grid = grid.copy()
                new_grid[py, px] = 5
                return new_grid
        return grid

    elif action == 7:
        # Action 7: Move player one step DOWN-RIGHT (diagonal)
        player_pos = np.argwhere(grid == 0)
        if len(player_pos) == 0:
            return grid
        py, px = player_pos[0]
        if py < H - 1 and px < W - 1:
            new_grid = grid.copy()
            new_grid[py, px] = 5
            new_grid[py + 1, px + 1] = 0
            return new_grid
        return grid

    return grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if the grid matches the win state pattern
    # Win state has:
    # - Row 0: all 5s
    # - Row 63: all 5s
    # - Rows 1-5, 58-62: 32 6s and 32 15s
    # - Rows 6-9, 14-17, 42-45: 6 6s, 24 5s, 2 6s, 2 15s, 24 5s, 6 15s
    # - Rows 10-13, 18-21, 22-25, 26-37, 54-57: complex pattern of 6s, 5s, 8s, 15s
    
    # Simplified check:
    # Check if row 0 and row 63 are all 5s
    if not np.all(grid[0] == 5) or not np.all(grid[63] == 5):
        return False
    
    # Check if rows 1-5 and 58-62 have the pattern
    for r in range(1, 6):
        if not (np.sum(grid[r] == 6) == 32 and np.sum(grid[r] == 15) == 32):
            return False
    for r in range(58, 63):
        if not (np.sum(grid[r] == 6) == 32 and np.sum(grid[r] == 15) == 32):
            return False
    
    # Check if rows 6-9 and 14-17 and 42-45 have the pattern
    for r in range(6, 10):
        if not (np.sum(grid[r] == 6) == 6 and np.sum(grid[r] == 5) == 24 and np.sum(grid[r] == 15) == 6):
            return False
    for r in range(14, 18):
        if not (np.sum(grid[r] == 6) == 6 and np.sum(grid[r] == 5) == 24 and np.sum(grid[r] == 15) == 6):
            return False
    for r in range(42, 46):
        if not (np.sum(grid[r] == 6) == 6 and np.sum(grid[r] == 5) == 24 and np.sum(grid[r] == 15) == 6):
            return False
    
    # Check if rows 10-13 and 18-21 have the pattern
    for r in range(10, 14):
        if not (np.sum(grid[r] == 6) == 6 and np.sum(grid[r] == 5) == 16 and np.sum(grid[r] == 15) == 6):
            return False
    for r in range(18, 22):
        if not (np.sum(grid[r] == 6) == 6 and np.sum(grid[r] == 5) == 8 and np.sum(grid[r] == 15) == 18):
            return False
    
    # Check if rows 22-25 have the pattern
    for r in range(22, 26):
        if not (np.sum(grid[r] == 6) == 6 and np.sum(grid[r] == 5) == 8 and np.sum(grid[r] == 15) == 18):
            return False
    
    # Check if rows 26-37 have the pattern
    for r in range(26, 38):
        if not (np.sum(grid[r] == 6) == 6 and np.sum(grid[r] == 5) == 20 and np.sum(grid[r] == 15) == 2):
            return False
    
    # Check if rows 46-53 have the pattern
    for r in range(46, 54):
        if not (np.sum(grid[r] == 6) == 6 and np.sum(grid[r] == 5) == 52 and np.sum(grid[r] == 15) == 6):
            return False
    
    # Check if rows 54-57 have the pattern
    for r in range(54, 58):
        if not (np.sum(grid[r] == 6) == 6 and np.sum(grid[r] == 5) == 8 and np.sum(grid[r] == 15) == 2):
            return False
    
    return True