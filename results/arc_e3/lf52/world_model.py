def engine(grid, action, data):
    """
    Simulate the effect of an action on the grid.
    grid: 2D list of integers (0=empty, 1=wall, 2=player, 3=goal)
    action: 0=up, 1=down, 2=left, 3=right
    data: dict with 'x', 'y' keys for player position, or None
    Returns: 2D list of the grid after the action
    """
    if data is None:
        return grid

    # Extract player position
    x = data['x']
    y = data['y']

    # Determine the new position based on the action
    new_x, new_y = x, y
    if action == 0:  # Up
        new_y -= 1
    elif action == 1:  # Down
        new_y += 1
    elif action == 2:  # Left
        new_x -= 1
    elif action == 3:  # Right
        new_x += 1

    # Check if the new position is within the grid boundaries
    if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]):
        # Check if the new position is a wall
        if grid[new_y][new_x] == 1:
            # If it's a wall, the player stays in the current position
            return grid

        # If it's not a wall, update the grid
        # The player moves to the new position
        grid[new_y][new_x] = 2
        grid[y][x] = 0

    return grid

def is_level_complete(grid):
    """
    Check if the level is complete.
    grid: 2D list of integers (0=empty, 1=1=wall, 2=player, 3=goal)
    Returns: True if the level is complete, False otherwise
    """
    # Check if the player is on the goal
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 2 and grid[i][j] == 3:
                return True
    return False