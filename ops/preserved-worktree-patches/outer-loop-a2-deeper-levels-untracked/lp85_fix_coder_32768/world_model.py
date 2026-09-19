import numpy as np

def engine(grid, action, data):
    # Define the rules for each action
    if action == 1:  # Up
        # Implement up movement logic
        pass
    elif action == 2:  # Down
        # Implement down movement logic
        pass
    elif action == 3:  # Left
        # Implement left movement logic
        pass
    elif action == 4:  # Right
        # Implement right movement logic
        pass
    elif action == 5:  # Toggle
        # Implement toggle logic
        pass
    elif action == 6:  # Click
        # Implement click logic
        px, py = data['x'], data['y']
        # Example: Change the clicked cell to a specific value
        grid[py, px] = 10
    elif action == 7:  # Special
        # Implement special action logic
        pass
    
    return grid

def is_level_complete(grid):
    # Define the conditions for a level to be complete
    # Example: Check if all cells with value 14 are replaced with 5
    return np.all(grid == 5)