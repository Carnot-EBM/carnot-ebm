def engine(grid, action, data):
    """
    Simulate one step of the game.
    Returns the next grid state after applying the action.
    """
    # Determine grid dimensions
    # Assuming the grid is a 1D list of integers representing an 8x8 grid
    # based on the error "wrong shape (8, 8)" which implies the input was expected
    # to be 8x8 but the code treated it as 2D lists.
    # We will treat the input as a 1D list and return a 1D list.
    
    length = len(grid)
    if length == 0:
        return grid
    
    # Assuming 8x8 grid based on the error message
    rows = 8
    cols = 8
    
    # Find player position (1)
    player_pos = None
    for i in range(length):
        if grid[i] == 1:
            player_pos = i
            break
    
    if player_pos is None:
        return grid
    
    # Find goal position (2)
    goal_pos = None
    for i in range(length):
        if grid[i] == 2:
            goal_pos = i
            break
    
    # Direction vectors for movement (Up, Down, Left, Right)
    # 0=Up, 1=Down, 2=Left, 3=Right
    # Mapping to (row_change, col_change)
    directions = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1)    # Right
    }
    
    # Calculate new position
    new_row = (player_pos // cols) + directions[action][0]
    new_col = (player_pos % cols) + directions[action][1]
    
    # Check bounds
    if 0 <= new_row < rows and 0 <= new_col < cols:
        new_pos = new_row * cols + new_col
        
        # Check for walls (0)
        # If the new position is a wall, stay in place
        if grid[new_pos] == 0:
            pass
        else:
            # Move player
            new_grid = grid[:]
            new_grid[player_pos] = 0
            new_grid[new_pos] = 1
            return new_grid
    
    return grid

def is_level_complete(grid):
    """
    Check if the level is complete.
    Returns True if the player has reached the goal.
    """
    length = len(grid)
    if length == 0:
        return False
    
    # Find player position (1)
    player_pos = None
    for i in range(length):
        if grid[i] == 1:
            player_pos = i
            break
    
    if player_pos is None:
        return False
    
    # Find goal position (2)
    goal_pos = None
    for i in range(length):
        if grid[i] == 2:
            goal_pos = i
            break
    
    return player_pos == goal_pos