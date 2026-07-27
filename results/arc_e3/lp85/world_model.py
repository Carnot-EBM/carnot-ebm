def engine(grid, action, data):
    """
    Simulates one step of the game.
    - grid: 2D list of integers (the current state).
    - action: integer 0..7 (0=Up, 1=Down, 2=Left, 3=Right, 4=Up-Left, 5=Up-Right, 6=Down-Left, 7=Down-Right).
    - data: dict with keys 'x' and 'y' (player coordinates).
    Returns a new grid with the player moved and the environment updated.
    """
    import copy
    new_grid = copy.deepcopy(grid)
    x, y = data['x'], data['y']
    dx, dy = 0, 0
    if action == 0: dx, dy = -1, 0
    elif action == 1: dx, dy = 1, 0
    elif action == 2: dx, dy = 0, -1
    elif action == 3: dx, dy = 0, 1
    elif action == 4: dx, dy = -1, -1
    elif action == 5: dx, dy = -1, 1
    elif action == 6: dx, dy = 1, -1
    elif action == 7: dx, dy = 1, 1

    nx, ny = x + dx, y + dy
    if 0 <= nx < len(new_grid) and 0 <= ny < len(new_grid[0]):
        new_grid[nx][ny] = 1
    else:
        new_grid[x][y] = 1

    return new_grid

def is_level_complete(grid):
    """
    Returns True if the level is complete.
    """
    return all(cell == 0 for row in grid for cell in row)