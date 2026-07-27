def engine(grid, action, data):
    """
    Simulates the effect of an action on the grid.
    
    Args:
        grid: The current grid state.
        action: The action to perform (0-7).
        data: A dictionary containing 'x' and 'y' coordinates.
    
    Returns:
        The updated grid after the action.
    """
    x = data['x']
    y = data['y']
    
    # Determine the direction based on the action
    # 0: Up, 1: Down, 2: Left, 3: Right, 4: Up-Left, 5: Up-Right, 6: Down-Left, 7: Down-Right
    dx = 0
    dy = 0
    
    if action == 0:
        dy = -1
    elif action == 1:
        dy = 1
    elif action == 2:
        dx = -1
    elif action == 3:
        dx = 1
    elif action == 4:
        dx = -1
        dy = -1
    elif action == 5:
        dx = 1
        dy = -1
    elif action == 6:
        dx = -1
        dy = 1
    elif action == 7:
        dx = 1
        dy = 1
    
    # Create a copy of the grid to store changes
    new_grid = [row[:] for row in grid]
    
    # Apply the action to the grid
    # The action moves the player and collects items in the direction of movement
    # We need to check the cells in the direction of movement and update them
    
    # Determine the range of cells to check
    # For diagonal moves, we check both horizontal and vertical cells
    # For cardinal moves, we check only the cardinal cells
    
    # Create a set of directions to check based on the action
    directions_to_check = []
    if action in [0, 1, 2, 3]:
        directions_to_check = [(dx, dy)]
    else:
        directions_to_check = [(dx, dy), (dx, 0), (0, dy)]
    
    # Check each direction and update the grid
    for ddx, ddy in directions_to_check:
        cx, cy = x + ddx, y + ddy
        
        # Check if the cell is within bounds
        if 0 <= cx < len(grid) and 0 <= cy < len(grid[0]):
            # Check if the cell contains an item
            if grid[cy][cx] != 0:
                # Update the cell with the item value
                new_grid[cy][cx] = grid[cy][cx]
                # Mark the cell as visited
                new_grid[cy][cx] = grid[cy][cx]
    
    return new_grid

def is_level_complete(grid):
    """
    Checks if the level is complete.
    """
    # Check if all items have been collected
    for row in grid:
        for cell in row:
            if cell != 0:
                return False
    
    return True