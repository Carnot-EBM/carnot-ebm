import numpy as np

def engine(grid, action, data):
    if action == 1:
        # Action 1: Move player (color 0) down (row+1)
        # Find player position
        player_pos = np.argwhere(grid == 0)
        if len(player_pos) == 0:
            return grid
        y, x = player_pos[0]
        new_grid = grid.copy()
        # Move player down
        if y + 1 < grid.shape[0]:
            new_grid[y + 1, x] = 0
            new_grid[y, x] = 5
        return new_grid
    elif action == 2:
        # Action 2: Move player (color 0) up (row-1)
        player_pos = np.argwhere(grid == 0)
        if len(player_pos) == 0:
            return grid
        y, x = player_pos[0]
        new_grid = grid.copy()
        if y - 1 >= 0:
            new_grid[y - 1, x] = 0
            new_grid[y, x] = 5
        return new_grid
    elif action == 3:
        # Action 3: Move player (color 0) right (col+1)
        player_pos = np.argwhere(grid == 0)
        if len(player_pos) == 0:
            return grid
        y, x = player_pos[0]
        new_grid = grid.copy()
        if x + 1 < grid.shape[1]:
            new_grid[y, x + 1] = 0
            new_grid[y, x] = 5
        return new_grid
    elif action == 4:
        # Action 4: Move player (color 0) left (col-1)
        player_pos = np.argwhere(grid == 0)
        if len(player_pos) == 0:
            return grid
        y, x = player_pos[0]
        new_grid = grid.copy()
        if x - 1 >= 0:
            new_grid[y, x - 1] = 0
            new_grid[y, x] = 5
        return new_grid
    elif action == 5:
        # Action 5: Move player (color 0) down-right (row+1, col+1)
        player_pos = np.argwhere(grid == 0)
        if len(player_pos) == 0:
            return grid
        y, x = player_pos[0]
        new_grid = grid.copy()
        if y + 1 < grid.shape[0] and x + 1 < grid.shape[1]:
            new_grid[y + 1, x + 1] = 0
            new_grid[y, x] = 5
        return new_grid
    elif action == 6:
        # Action 6: Click (data contains x, y)
        if data is None:
            return grid
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        # Convert pixel to logical (divide by 1)
        y, x = py, px
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            new_grid[y, x] = 5
        return new_grid
    elif action == 7:
        # Action 7: Move player (color 0) down-left (row+1, col-1)
        player_pos = np.argwhere(grid == 0)
        if len(player_pos) == 0:
            return grid
        y, x = player_pos[0]
        new_grid = grid.copy()
        if y + 1 < grid.shape[0] and x - 1 >= 0:
            new_grid[y + 1, x - 1] = 0
            new_grid[y, x] = 5
        return new_grid
    return grid

def is_level_complete(grid):
    # Check if grid matches win state pattern
    # Win state has specific structure:
    # - Row 0: all 5s
    # - Row 63: all 5s
    # - Rows 1-62: alternating 6 and 15
    # - Specific pattern in middle rows
    # Check if grid matches the win state pattern
    if grid.shape != (64, 64):
        return False
    
    # Check row 0 and 63
    if not np.all(grid[0] == 5) or not np.all(grid[63] == 5):
        return False
    
    # Check rows 1-62
    for i in range(1, 63):
        if not np.all(grid[i] == 6) and not np.all(grid[i] == 15):
            return False
    
    # Check specific pattern in middle rows
    # Rows 6-9, 14-17, 18-21, 22-25, 26-37, 38-41, 42-45, 46-53, 54-57
    # These rows have specific patterns with 6, 5, 8, 15
    for i in range(6, 58):
        if i in [6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]:
            # Check if row matches expected pattern
            # Pattern: 6x6, 5x24, 6x2, 15x2, 5x24, 15x6
            if i in [6, 7, 8, 9]:
                expected = np.array([6]*6 + [5]*24 + [6]*2 + [15]*2 + [5]*24 + [15]*6)
                if not np.array_equal(grid[i], expected):
                    return False
            elif i in [14, 15, 16, 17]:
                expected = np.array([6]*6 + [5]*24 + [6]*2 + [15]*2 + [5]*24 + [15]*6)
                if not np.array_equal(grid[i], expected):
                    return False
            elif i in [18, 19, 20, 21]:
                expected = np.array([6]*6 + [5]*8 + [6]*18 + [15]*18 + [5]*8 + [15]*6)
                if not np.array_equal(grid[i], expected):
                    return False
            elif i in [22, 23, 24, 25]:
                expected = np.array([6]*6 + [5]*8 + [6]*4 + [5]*4 + [6]*10 + [15]*18 + [5]*8 + [15]*6)
                if not np.array_equal(grid[i], expected):
                    return False
            elif i in [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]:
                expected = np.array([6]*6 + [5]*20 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [6]*2 + [15]*2 + [5]*24 + [15]*6)
                if not np.array_equal(grid[i], expected):
                    return False
            elif i in [38, 39, 40, 41]:
                expected = np.array([6]*6 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*5 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [6]*2 + [15]*2 + [5]*5 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [15]*6)
                if not np.array_equal(grid[i], expected):
                    return False
            elif i in [42, 43, 44, 45]:
                expected = np.array([6]*6 + [5]*24 + [6]*2 + [15]*2 + [5]*24 + [15]*6)
                if not np.array_equal(grid[i], expected):
                    return False
            elif i in [46, 47, 48, 49, 50, 51, 52, 53]:
                expected = np.array([6]*6 + [5]*52 + [15]*6)
                if not np.array_equal(grid[i], expected):
                    return False
            elif i in [54, 55, 56, 57]:
                expected = np.array([6]*6 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [8]*1 + [5]*1 + [15]*6)
                if not np.array_equal(grid[i], expected):
                    return False
    
    return True