import numpy as np

def engine(grid, action, data):
    if action == 1:
        return apply_action_1(grid, data)
    elif action == 2:
        return apply_action_2(grid, data)
    elif action == 3:
        return apply_action_3(grid, data)
    elif action == 4:
        return apply_action_4(grid, data)
    elif action == 5:
        return apply_action_5(grid, data)
    elif action == 6:
        return apply_action_6(grid, data)
    elif action == 7:
        return apply_action_7(grid, data)
    else:
        return grid.copy()

def apply_action_1(grid, data):
    h, w = grid.shape
    new_grid = grid.copy()
    # Action 1: Move player down (gravity-like)
    # Player is at (49, 19) and (49, 39) based on initial grid
    # Move player down by 1
    # Player color is 5
    # Apply gravity to player
    for y in range(h - 1, -1, -1):
        for x in range(w):
            if new_grid[y, x] == 5:
                if y + 1 < h and new_grid[y + 1, x] == 0:
                    new_grid[y, x] = 0
                    new_grid[y + 1, x] = 5
    return new_grid

def apply_action_2(grid, data):
    h, w = grid.shape
    new_grid = grid.copy()
    # Action 2: Move player up
    # Player is at (49, 19) and (49, 39)
    # Move player up by 1
    for y in range(h):
        for x in range(w):
            if new_grid[y, x] == 5:
                if y - 1 >= 0 and new_grid[y - 1, x] == 0:
                    new_grid[y, x] = 0
                    new_grid[y - 1, x] = 5
    return new_grid

def apply_action_3(grid, data):
    h, w = grid.shape
    new_grid = grid.copy()
    # Action 3: Move player left
    # Player is at (49, 19) and (49, 39)
    # Move player left by 1
    for y in range(h):
        for x in range(w):
            if new_grid[y, x] == 5:
                if x - 1 >= 0 and new_grid[y, x - 1] == 0:
                    new_grid[y, x] = 0
                    new_grid[y, x - 1] = 5
    return new_grid

def apply_action_4(grid, data):
    h, w = grid.shape
    new_grid = grid.copy()
    # Action 4: Move player right
    # Player is at (49, 19) and (49, 39)
    # Move player right by 1
    for y in range(h):
        for x in range(w):
            if new_grid[y, x] == 5:
                if x + 1 < w and new_grid[y, x + 1] == 0:
                    new_grid[y, x] = 0
                    new_grid[y, x + 1] = 5
    return new_grid

def apply_action_5(grid, data):
    h, w = grid.shape
    new_grid = grid.copy()
    # Action 5: Toggle cell (0 to 1)
    # Toggle cells at (0, 63) and (63, 0)
    new_grid[0, 63] = 1 - new_grid[0, 63]
    new_grid[63, 0] = 1 - new_grid[63, 0]
    return new_grid

def apply_action_6(grid, data):
    h, w = grid.shape
    new_grid = grid.copy()
    # Action 6: Click at pixel coordinates
    if data and 'x' in data and 'y' in data:
        px, py = data['x'], data['y']
        # Convert pixel to logical
        lx, ly = px // 1, py // 1
        if 0 <= ly < h and 0 <= lx < w:
            new_grid[ly, lx] = 1 - new_grid[ly, lx]
    return new_grid

def apply_action_7(grid, data):
    h, w = grid.shape
    new_grid = grid.copy()
    # Action 7: No-op or special action
    return new_grid

def is_level_complete(grid):
    # Check if the level is complete
    # Based on the initial grid, the level is complete when the player reaches the goal
    # The goal is likely at (63, 63) or similar
    # Check if the player is at the goal
    h, w = grid.shape
    for y in range(h):
        for x in range(w):
            if grid[y, x] == 5:
                if y == 63 and x == 63:
                    return True
    return False