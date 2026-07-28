def engine(grid, action, data):
    """
    Simulates the world model transition.
    grid: 2D list of integers (the world state)
    action: integer (0-7) representing direction
    data: dict with 'x', 'y' (agent position)
    Returns: new grid after applying action
    """
    # Create a copy of the grid to avoid modifying the original
    new_grid = [row[:] for row in grid]
    
    # Get the agent's position
    x = data['x']
    y = data['y']
    
    # Map action to direction deltas
    # Based on the failing case, action 6 (up) causes a vertical shift
    # The grid appears to be a 3D-like structure flattened or represented as 2D with height
    # Looking at the true_change, it seems like the grid is being modified in a specific way
    
    # Let's analyze the pattern from the failing case:
    # Action 6 (up) at position (1, 17)
    # The true_change shows a pattern of values being shifted
    
    # Based on the pattern, it seems like the grid is being manipulated in a way that
    # creates a "wave" or "shift" effect. The values are being moved in a specific pattern.
    
    # Let's try to understand the grid structure from the true_change
    # The true_change has 8 rows, each with 4 columns
    # This suggests the grid might be 8x4 or similar
    
    # Looking at the pattern, it seems like the grid is being shifted vertically
    # with some wrapping or boundary handling
    
    # Let's implement a simple shift operation based on the action
    # Action 6 (up) seems to shift the grid up
    
    # For action 6 (up), we shift the grid up
    if action == 6:
        # Shift up: move each row to the previous row, with the first row becoming the last
        # But looking at the true_change, it seems more complex
        # Let's try a different approach - maybe it's a wave effect
        
        # Based on the pattern, it seems like the grid is being modified in a way that
        # creates a specific pattern of changes
        
        # Let's try to implement the shift based on the observed pattern
        # The pattern shows that certain rows are being shifted
        
        # For now, let's implement a simple shift and see if it matches
        # Shift up by 1 row
        if len(new_grid) > 1:
            new_grid.insert(0, new_grid[-1])  # Move last row to top
            new_grid.pop()  # Remove last row
            
    elif action == 7:
        # Shift down
        if len(new_grid) > 1:
            new_grid.append(new_grid[0])  # Move first row to bottom
            new_grid.pop(0)  # Remove first row
            
    elif action == 1:
        # Shift right
        if len(new_grid) > 0 and len(new_grid[0]) > 0:
            new_grid[0].insert(0, new_grid[0][-1])
            new_grid[0].pop()
            
    elif action == 3:
        # Shift left
        if len(new_grid) > 0 and len(new_grid[0]) > 0:
            new_grid[0].pop(0)
            new_grid[0].insert(0, new_grid[0][-1])
            
    # For other actions, we might need to implement similar logic
    # But based on the failing case, we need to handle action 6 specifically
    
    return new_grid

def is_level_complete(grid):
    """
    Checks if the level is complete.
    grid: 2D list of integers
    Returns: boolean indicating if the level is complete
    """
    # Check if all cells in the grid are non-zero
    # This is a simple heuristic for level completion
    for row in grid:
        for cell in row:
            if cell == 0:
                return False
    return True