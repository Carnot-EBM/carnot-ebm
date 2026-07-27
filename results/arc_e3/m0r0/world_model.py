def engine(grid, action, data):
    """
    Simulates the effect of an action on the grid.
    grid: 2D list of integers representing the current state.
    action: integer action code (0=up, 1=down, 2=left, 3=right, 4=stay, 5=rotate).
    data: unused.
    Returns: new grid after applying the action.
    """
    if action == 5:
        return rotate_grid(grid)
    elif action == 0:
        return move_grid(grid, 0)
    elif action == 1:
        return move_grid(grid, 1)
    elif action == 2:
        return move_grid(grid, 2)
    elif action == 3:
        return move_grid(grid, 3)
    elif action == 4:
        return grid
    else:
        return grid

def rotate_grid(grid):
    """
    Rotates the grid 90 degrees clockwise.
    """
    rows = len(grid)
    cols = len(grid[0])
    new_grid = [[0 for _ in range(rows)] for _ in range(cols)]
    for r in range(rows):
        for c in range(cols):
            new_grid[c][rows - 1 - r] = grid[r][c]
    return new_grid

def move_grid(grid, direction):
    """
    Moves all non-zero elements in the grid in the specified direction.
    direction: 0=up, 1=down, 2=left, 3=right.
    """
    rows = len(grid)
    cols = len(grid[0])
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    if direction == 0:  # Up
        for c in range(cols):
            col_vals = [grid[r][c] for r in range(rows) if grid[r][c] != 0]
            for r in range(rows):
                if r < len(col_vals):
                    new_grid[r][c] = col_vals[r]
                else:
                    new_grid[r][c] = 0
    elif direction == 1:  # Down
        for c in range(cols):
            col_vals = [grid[r][c] for r in range(rows) if grid[r][c] != 0]
            for r in range(rows - 1, -1, -1):
                if r >= len(col_vals):
                    new_grid[r][c] = 0
                else:
                    new_grid[r][c] = col_vals[len(col_vals) - 1 - (rows - 1 - r)]
    elif direction == 2:  # Left
        for r in range(rows):
            row_vals = [grid[r][c] for c in range(cols) if grid[r][c] != 0]
            for c in range(cols):
                if c < len(row_vals):
                    new_grid[r][c] = row_vals[c]
                else:
                    new_grid[r][c] = 0
    elif direction == 3:  # Right
        for r in range(rows):
            row_vals = [grid[r][c] for c in range(cols) if grid[r][c] != 0]
            for c in range(cols):
                if c < len(row_vals):
                    new_grid[r][cols - 1 - c] = row_vals[c]
                else:
                    new_grid[r][cols - 1 - c] = 0
    return new_grid

def is_level_complete(grid):
    """
    Checks if the level is complete.
    Returns True if all non-zero cells in the grid are 0.
    """
    for row in grid:
        for cell in row:
            if cell != 0:
                return False
    return True