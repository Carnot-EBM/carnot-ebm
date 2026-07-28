def engine(grid, action, data):
    """
    Simulates one step of the game.
    
    Args:
        grid: The current game grid (list of lists).
        action: The action to perform (0-7).
        data: Dictionary containing 'x' and 'y' coordinates of the player.
    
    Returns:
        The new grid after performing the action.
    """
    # Get player position
    x = data['x']
    y = data['y']
    
    # Get grid dimensions
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Create a copy of the grid to avoid modifying the original
    new_grid = [row[:] for row in grid]
    
    # Define action directions
    # 0: Up, 1: Down, 2: Left, 3: Right, 4: Up-Left, 5: Up-Right, 6: Down-Left, 7: Down-Right
    directions = [
        (-1, 0),  # 0: Up
        (1, 0),   # 1: Down
        (0, -1),  # 2: Left
        (0, 1),   # 3: Right
        (-1, -1), # 4: Up-Left
        (-1, 1),  # 5: Up-Right
        (1, -1),  # 6: Down-Left
        (1, 1)    # 7: Down-Right
    ]
    
    # Get the direction for the current action
    dx, dy = directions[action]
    
    # Calculate new position
    new_x = x + dx
    new_y = y + dy
    
    # Check if the new position is within bounds
    if 0 <= new_x < rows and 0 <= new_y < cols:
        # Player moved to a new position
        # Clear the old position
        new_grid[x][y] = 0
        
        # Set the new position
        new_grid[new_x][new_y] = 1
        
        # Update the player's position in the grid
        # The grid now represents the player's position
        
        # Check if the player has reached the goal
        # The goal is at position (rows-1, cols-1)
        if new_x == rows - 1 and new_y == cols - 1:
            # Level complete
            return new_grid
    
    # If the player is out of bounds, the grid remains unchanged
    return new_grid

def is_level_complete(grid):
    """
    Checks if the level is complete.
    
    Args:
        grid: The current game grid (list of lists).
    
    Returns:
        True if the level is complete, False otherwise.
    """
    # Check if the player has reached the goal
    # The goal is at position (rows-1, cols-1)
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Check if the player is at the goal position
    if rows > 0 and cols > 0:
        if grid[rows-1][cols-1] == 1:
            return True
    
    return False