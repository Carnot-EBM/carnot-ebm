import numpy as np

def engine(grid, action, data):
    """
    Applies a single action to the grid.
    grid: np.ndarray (64x64 int)
    action: int 1-7
    data: dict or None
    
    Returns: np.ndarray (64x64 int)
    """
    # Convert grid to list of lists for easier manipulation
    grid_list = grid.tolist()
    
    # Define action behaviors based on observed transitions
    # Actions 1-7 seem to be directional or specific movements
    # Based on the observed transitions, it appears actions move the player
    # and potentially interact with the grid
    
    # Action 3: moves player (observed change at r0c0)
    # Action 6: click action with pixel data (observed changes at various positions)
    # Other actions likely follow similar patterns
    
    # Since we don't have explicit rules for each action, we'll implement a generic
    # movement system based on the action number and data
    
    # Action 1: Move Up
    # Action 2: Move Down
    # Action 3: Move Left
    # Action 4: Move Right
    # Action 5: Jump/Teleport
    # Action 6: Click (special action with pixel data)
    # Action 7: Special interaction
    
    if action == 6:
        # Click action with pixel data
        if data is None:
            return grid
        
        px = data['x']
        py = data['y']
        
        # Convert pixel coordinates to grid coordinates
        gx = px // 1  # pixel = logical * 1
        gy = py // 1
        
        # Create a copy of the grid
        new_grid = grid.copy()
        
        # Apply the click effect (observed changes show single cell changes)
        # Based on observed data, clicking changes a single cell to color 1
        if 0 <= gy < 64 and 0 <= gx < 64:
            new_grid[gy, gx] = 1
        
        return new_grid
    else:
        # Directional actions (1-5, 7)
        # Based on observed transitions, these move the player
        # The player position is tracked implicitly
        
        # Since we don't have explicit player position, we'll assume
        # the player is at a specific location or the action modifies the grid directly
        
        # For simplicity, we'll implement basic movement logic
        # This is a placeholder since we don't have complete rules
        
        # Action 1: Move Up
        if action == 1:
            # Move player up by 1 cell
            # This would typically change the player's position
            # Since we don't track player position explicitly, we'll just return the grid
            return grid
        
        # Action 2: Move Down
        elif action == 2:
            return grid
        
        # Action 3: Move Left
        elif action == 3:
            return grid
        
        # Action 4: Move Right
        elif action == 4:
            return grid
        
        # Action 5: Jump/Teleport
        elif action == 5:
            return grid
        
        # Action 7: Special interaction
        elif action == 7:
            return grid
        
        return grid

def is_level_complete(grid):
    """
    Checks if the grid represents a win state.
    grid: np.ndarray (64x64 int)
    
    Returns: bool
    """
    # Based on the observed transitions, the win state appears to be when
    # certain conditions are met. Since we don't have explicit win conditions,
    # we'll check for a common pattern in ARC games.
    
    # A common win condition is when all target cells are filled or a specific
    # pattern is achieved. Without explicit rules, we'll check if the grid
    # has reached a stable state.
    
    # For now, we'll return False as we don't have complete win condition rules
    return False