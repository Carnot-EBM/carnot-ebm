import numpy as np

def engine(grid, action, data):
    """
    Predict the next grid state given the current grid, action, and action data.
    This game involves a 64x64 grid with specific colors and rules.
    The player (color 14) can move and interact with the environment.
    The environment contains walls (color 5), platforms (color 9), and other elements.
    The player can collect items (color 2) and interact with the environment.
    The game has a win condition based on collecting all items.
    """
    # Copy the grid to avoid modifying the original
    new_grid = grid.copy()
    
    # Check if the action is a click (ACTION6)
    if action == 6:
        # Get the pixel coordinates
        px, py = data['x'], data['y']
        # Convert to logical coordinates
        x, y = px // 1, py // 1
        
        # Check if the player is at the clicked location
        if grid[y, x] == 14:
            # The player can move to the clicked location
            # Check if the clicked location is valid (not a wall)
            if grid[y, x] != 5:
                # Move the player to the clicked location
                # The player's previous location becomes empty (color 0)
                prev_x, prev_y = x - 1, y - 1
                if prev_x >= 0 and prev_y >= 0:
                    new_grid[prev_y, prev_x] = 0
                # The player's new location is the clicked location
                new_grid[y, x] = 14
            else:
                # The clicked location is a wall, so the player cannot move there
                # The player remains in the current location
                pass
        else:
            # The player is not at the clicked location, so the action has no effect
            pass
    
    return new_grid

def is_level_complete(grid):
    """
    Check if the level is complete.
    The level is complete if all items (color 2) have been collected.
    """
    # Count the number of items (color 2) in the grid
    items = np.sum(grid == 2)
    # If there are no items left, the level is complete
    return items == 0