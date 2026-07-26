def engine(grid, action, data):
    """
    Simulate one step of the game.
    - grid: 2D list of integers (0-63)
    - action: 0=up, 1=down, 2=left, 3=right, 4=rotate
    - data: None (unused)
    Returns: new grid after applying action
    """
    h, w = len(grid), len(grid[0])
    new_grid = [row[:] for row in grid]
    
    # Handle rotation (action 4)
    if action == 4:
        new_grid = [list(reversed([grid[h-1-i][j] for i in range(h)])) for j in range(w)]
        return new_grid
    
    # Handle movement (actions 0-3)
    for i in range(h):
        for j in range(w):
            val = grid[i][j]
            if val == 0:
                continue
            
            # Determine movement direction based on action
            di, dj = 0, 0
            if action == 0:  # Up
                di, dj = -1, 0
            elif action == 1:  # Down
                di, dj = 1, 0
            elif action == 2:  # Left
                di, dj = 0, -1
            elif action == 3:  # Right
                di, dj = 0, 1
            
            # Try to move
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and grid[ni][nj] == 0:
                new_grid[ni][nj] = val
                new_grid[i][j] = 0
            # If blocked, stay in place (no change)
    
    return new_grid

def is_level_complete(grid):
    """
    Check if the level is complete.
    Returns True if all non-zero cells are in their final positions.
    """
    h, w = len(grid), len(grid[0])
    for i in range(h):
        for j in range(w):
            if grid[i][j] != 0:
                # Check if this cell is in a final position
                # For simplicity, we consider a cell final if it's not in the first row
                if i == 0:
                    return False
    return True