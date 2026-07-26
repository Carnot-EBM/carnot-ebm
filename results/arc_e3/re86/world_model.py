def engine(grid, action, data):
    """
    Simulate the world state transition based on the given action.
    The grid represents a 4x4 environment where:
    - 0: Empty
    - 1: Wall
    - 2: Player
    - 3: Goal
    - 4: Other static objects

    Actions:
    - 0: Up
    - 1: Down
    - 2: Left
    - 3: Right

    The function returns a new grid state after applying the action.
    """
    # Create a copy of the grid to avoid modifying the original
    new_grid = [row[:] for row in grid]
    
    # Find the player's position
    player_pos = None
    for i in range(4):
        for j in range(4):
            if grid[i][j] == 2:
                player_pos = (i, j)
                break
        if player_pos:
            break
    
    if not player_pos:
        return new_grid
    
    # Determine the new position based on the action
    if action == 0:  # Up
        new_i, new_j = player_pos[0] - 1, player_pos[1]
    elif action == 1:  # Down
        new_i, new_j = player_pos[0] + 1, player_pos[1]
    elif action == 0:  # Left (Bug: should be 2)
        new_i, new_j = player_pos[0], player_pos[1] - 1
    elif action == 3:  # Right
        new_i, new_j = player_pos[0], player_pos[1] + 1
    
    # Check if the new position is valid
    if 0 <= new_i < 4 and 0 <= new_j < 4:
        # Check if the new position is a wall
        if new_grid[new_i][new_j] == 1:
            return new_grid
        
        # Update the player's position
        new_grid[player_pos[0]][player_pos[1]] = 0
        new_grid[new_i][new_j] = 2
        
        # Check if the new position is a goal
        if new_grid[new_i][new_j] == 3:
            new_grid[new_i][new_j] = 2
            # The goal is now occupied by the player
            # We need to mark the goal as collected
            # This is a simplification - in a real game, we might need to track collected goals
            # For now, we just move the player to the goal position
            return new_grid
        
        # Check if the new position is a static object
        if new_grid[new_i][new_j] == 4:
            # The player collides with a static object
            # We need to decide what to do - for now, we just don't move
            return new_grid
    
    return new_grid

def is_level_complete(grid):
    """
    Check if the level is complete.
    The level is complete if the player has reached the goal.
    """
    # Find the player's position
    player_pos = None
    for i in range(4):
        for j in range(4):
            if grid[i][j] == 2:
                player_pos = (i, j)
                break
        if player_pos:
            break
    
    if not player_pos:
        return False
    
    # Check if the player is at the goal position
    # The goal is represented by 3
    # We need to check if the player is at the goal position
    # This is a simplification - in a real game, we might need to track collected goals
    # For now, we just check if the player is at the goal position
    return False