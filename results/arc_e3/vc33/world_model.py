def engine(grid, action, data):
    """
    Simulates one step of the world.
    - grid: 2D array of ints (0=empty, 1=wall, 2=agent, 3=goal, 4=goal_reached, 5=goal_reached, 6=goal_reached, 7=goal_reached, 8=goal_reached, 9=goal_reached)
    - action: 0=up, 1=down, 2=left, 3=right, 4=stay, 5=up, 6=down, 7=left, 8=right, 9=stay
    - data: {"x": int, "y": int}
    Returns: 2D array of ints (same shape as grid)
    """
    # Find agent position
    agent_x = data["x"]
    agent_y = data["y"]
    
    # Determine movement direction
    if action == 0 or action == 5:  # up
        new_x = agent_x
        new_y = agent_y - 1
    elif action == 1 or action == 6:  # down
        new_x = agent_x
        new_y = agent_y + 1
    elif action == 2 or action == 7:  # left
        new_x = agent_x - 1
        new_y = agent_y
    elif action == 3 or action == 8:  # right
        new_x = agent_x + 1
        new_y = agent_y
    else:  # stay
        new_x = agent_x
        new_y = agent_y
    
    # Check bounds and collisions
    if new_x < 0 or new_x >= len(grid[0]) or new_y < 0 or new_y >= len(grid):
        # Out of bounds - stay in place
        new_x = agent_x
        new_y = agent_y
    
    # Check for walls
    if grid[new_y][new_x] == 1:
        # Hit wall - stay in place
        new_x = agent_x
        new_y = agent_y
    
    # Create new grid
    new_grid = [row[:] for row in grid]
    
    # Clear old agent position
    new_grid[agent_y][agent_x] = 0
    
    # Place agent at new position
    new_grid[new_y][new_x] = 2
    
    # Check for goal
    if grid[new_y][new_x] == 3:
        new_grid[new_y][new_x] = 4
    
    return new_grid

def is_level_complete(grid):
    """
    Checks if the level is complete.
    Returns: True if all goals are reached, False otherwise
    """
    # Count goals
    goal_count = 0
    for row in grid:
        for cell in row:
            if cell == 3:
                goal_count += 1
    
    # Count reached goals
    reached_count = 0
    for row in grid:
        for cell in row:
            if cell in [4, 5, 6, 7, 8, 9]:
                reached_count += 1
    
    # Level is complete if all goals are reached
    return goal_count == reached_count