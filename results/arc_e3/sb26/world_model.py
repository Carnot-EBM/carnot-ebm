def engine(grid, action, data):
    """
    Simulates the effect of an action on the grid.
    action: 0=up, 1=down, 2=left, 3=right, 4=jump, 5=shoot, 6=move
    data: {"x": int, "y": int}
    Returns: list of [new_x, new_y, new_z, new_w] for each affected cell
    """
    x, y = data["x"], data["y"]
    result = []
    
    # Action 6: Move (walk)
    if action == 6:
        # Check if the target cell is empty (value 0)
        target_x, target_y = x + 1, y
        if 0 <= target_x < len(grid[0]) and 0 <= target_y < len(grid):
            if grid[target_y][target_x] == 0:
                # Move the player
                grid[y][x] = 0
                grid[target_y][target_x] = 5
                result.append([target_x, target_y, 0, 5])
            # If target is not empty, do nothing (no change)
        # If out of bounds, do nothing
    
    # Action 5: Shoot
    elif action == 5:
        # Check if the target cell is empty (value 0)
        target_x, target_y = x + 1, y
        if 0 <= target_x < len(grid[0]) and 0 <= target_y < len(grid):
            if grid[target_y][target_x] == 0:
                # Shoot the cell
                grid[y][x] = 0
                grid[target_y][target_x] = 5
                result.append([target_x, target_y, 0, 5])
            # If target is not empty, do nothing (no change)
    
    # Action 4: Jump
    elif action == 4:
        # Check if the target cell is empty (value 0)
        target_x, target_y = x, y + 1
        if 0 <= target_x < len(grid[0]) and 0 <= target_y < len(grid):
            if grid[target_y][target_x] == 0:
                # Jump the player
                grid[y][x] = 0
                grid[target_y][target_x] = 5
                result.append([target_x, target_y, 0, 5])
            # If target is not empty, do nothing (no change)
    
    return result

def is_level_complete(grid):
    """
    Checks if the level is complete.
    Returns: True if all cells are 0 or 5, False otherwise
    """
    for row in grid:
        for cell in row:
            if cell != 0 and cell != 5:
                return False
    return True