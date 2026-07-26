def engine(grid, action, data):
    """
    Simulate the agent's movement and interaction with the environment.
    Handles movement, wall collisions, and object interactions.
    """
    x, y = data['x'], data['y']
    action_map = {
        0: (0, 0),
        1: (0, -1),
        2: (0, 1),
        3: (-1, 0),
        4: (1, 0),
        5: (-1, -1),
        6: (-1, 1),
        7: (1, -1),
        8: (1, 1)
    }
    dx, dy = action_map.get(action, (0, 0))
    new_x, new_y = x + dx, y + dy

    # Check for wall collisions
    if new_x < 0 or new_x >= len(grid) or new_y < 0 or new_y >= len(grid[0]):
        return grid

    # Check for wall collisions
    if grid[new_x][new_y] == 1:
        return grid

    # Check for object interactions
    # Refactored to handle all object types (2-8) uniformly
    if 2 <= grid[new_x][new_y] <= 8:
        # Create a copy of the grid to avoid modifying the original
        new_grid = [row[:] for row in grid]
        new_grid[new_x][new_y] = 0
        return new_grid

    # Update position
    # Create a copy of the grid to avoid modifying the original
    new_grid = [row[:] for row in grid]
    new_grid[new_x][new_y] = 1
    return new_grid

def is_level_complete(grid):
    """
    Check if the level is complete.
    Returns True if all objects have been collected.
    """
    for row in grid:
        for cell in row:
            if cell != 0:
                return False
    return True