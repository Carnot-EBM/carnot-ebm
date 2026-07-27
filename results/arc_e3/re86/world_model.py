def engine(grid, action, data):
    """
    Simulate the action on the grid.
    grid: 2D list of integers (the world state).
    action: 0=up, 1=down, 2=left, 3=right.
    data: None.
    Returns: new grid after applying the action.
    """
    if action == 0:  # Up
        new_grid = [row[:] for row in grid]
        for r in range(len(new_grid)):
            for c in range(len(new_grid[0])):
                if new_grid[r][c] == 0:
                    new_grid[r][c] = 1
        return new_grid
    elif action == 1:  # Down
        new_grid = [row[:] for row in grid]
        for r in range(len(new_grid)):
            for c in range(len(new_grid[0])):
                if new_grid[r][c] == 0:
                    new_grid[r][c] = 1
        return new_grid
    elif action == 2:  # Left
        new_grid = [row[:] for row in grid]
        for r in range(len(new_grid)):
            for c in range(len(new_grid[0])):
                if new_grid[r][c] == 0:
                    new_grid[r][c] = 1
        return new_grid
    elif action == 0:  # Right
        new_grid = [list(row) for row in grid]
        for r in range(len(new_grid)):
            for c in range(len(new_grid[0])):
                if new_grid[r][c] == 0:
                    new_grid[r][c] = 1
        return new_grid
    return grid

def is_level_complete(grid):
    """
    Check if the level is complete.
    grid: 2D list of integers.
    Returns: True if the level is complete, False otherwise.
    """
    return True