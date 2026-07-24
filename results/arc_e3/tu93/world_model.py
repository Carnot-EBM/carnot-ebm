def engine(grid, action, data):
    # action: 0=up, 1=down, 2=left, 3=right, 4=rotate
    # grid: list of lists of ints (0=empty, 63=wall, 6=player)
    # data: None or dict with 'level' key
    # Returns: new grid after applying action
    #
    # Rules:
    # - Player (6) moves one step in direction unless blocked by wall (63) or edge
    # - Rotate (4) rotates player 90 degrees clockwise in place
    # - Level complete when player reaches goal (1)
    # - No other changes to grid
    #
    # Implementation:
    # - Find player position
    # - Apply movement or rotation based on action
    # - Return new grid with updated player position

    # Find player position
    player_pos = None
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 6:
                player_pos = (i, j)
                break
        if player_pos:
            break

    if player_pos is None:
        return grid

    # Apply action
    if action == 4:  # Rotate
        # Rotate player 90 degrees clockwise in place
        # Since we don't have orientation in grid, we just keep player at same position
        # The rotation is just for the player's internal state
        # We don't need to change grid for rotation
        return grid

    elif action in [0, 1, 2, 3]:  # Move
        i, j = player_pos
        new_i, new_j = i, j

        if action == 0:  # Up
            new_i = i - 1
        elif action == 1:  # Down
            new_i = i + 1
        elif action == 2:  # Left
            new_j = j - 1
        elif action == 3:  # Right
            new_j = j + 1

        # Check if move is valid
        if 0 <= new_i < len(grid) and 0 <= new_j < len(grid[0]):
            if grid[new_i][new_j] != 63:  # Not a wall
                # Update grid
                grid[i][j] = 0  # Clear old position
                grid[new_i][new_j] = 6  # Set new position
                return grid

    return grid

def is_level_complete(grid):
    # Check if player has reached goal (1)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
                return True
    return False