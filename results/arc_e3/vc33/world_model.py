def engine(grid, action, data):
    """
    Simulates one action on the grid.
    - grid: 2D list of [x, y, z, w] (x=0..3, y=0..23, z=0..3, w=0..3)
    - action: 0=up, 1=down, 2=left, 3=right, 4=rotate, 5=rotate, 6=move
    - data: {"x": int, "y": int} current player position
    Returns: new grid after applying action
    """
    # Find player position
    px, py = data["x"], data["y"]
    
    # Determine new position based on action
    if action == 0:  # up
        new_x, new_y = px, py - 1
    elif action == 1:  # down
        new_x, new_y = px, py + 1
    elif action == 2:  # left
        new_x, new_y = px - 1, py
    elif action == 3:  # right
        new_x, new_y = px + 1, py
    elif action == 4:  # rotate
        new_x, new_y = px, py
    elif action == 5:  # rotate
        new_x, new_y = px, py
    elif action == 6:  # move
        new_x, new_y = px, py
    
    # Check bounds
    if new_x < 0 or new_x > 3 or new_y < 0 or new_y > 23:
        return grid
    
    # Create new grid
    new_grid = [row[:] for row in grid]
    
    # Remove block from old position
    new_grid[px][py] = [0, 0, 0, 0]
    
    # Add block to new position
    new_grid[new_x][new_y] = block
    
    return new_grid

def is_level_complete(grid):
    """
    Checks if the level is complete.
    Returns True if all blocks are in their final positions.
    """
    # Check if all blocks are in their final positions
    for row in grid:
        for block in row:
            if block != [0, 0, 0, 0] and block != [0, 0, 0, 0]:
                return False
    return True