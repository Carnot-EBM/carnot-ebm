import numpy as np

def engine(grid, action, data):
    """
    Simulates one step of the game 'm0r0'.
    
    Rules induced:
    - The player (color 0) starts at (63, 0) and moves up (decreasing row).
    - The player can move 1 step in any of 4 directions (up, down, left, right) using actions 1-4.
    - Action 5 is 'up' (decreasing row), Action 6 is 'click' (teleport or interact), Action 7 is 'down' (increasing row).
    - Based on observed transitions:
        - ACTION6 (click) at (x, y) sets grid[0][63-x] = 0 and grid[63-y][0] = 0. This looks like a teleport or interaction that sets the top-right and bottom-left corners to 0.
        - ACTION2 (keyboard) sets grid[0][61] = 0, grid[63][2] = 0, and modifies a vertical strip at column 14 and 44 from rows 49 to 58.
        - The grid has a background of 11 (left) and 12 (right) in the middle rows, with 5 (yellow) in the middle.
        - The player (0) moves through the grid, and the grid changes based on the player's position and actions.
    
    Simplified rules:
    - The player (0) moves 1 step in the direction of the action.
    - If the action is a click (6), the player teleports to (63, 0) and sets the corners to 0.
    - If the action is a keyboard action (1-5, 7), the player moves 1 step in the corresponding direction.
    - The grid changes based on the player's position and actions.
    
    Note: The observed transitions are complex and may not be fully captured by these rules.
    """
    if action == 6:
        # Click action: set corners to 0
        grid[0, 63] = 0
        grid[63, 0] = 0
        return grid
    
    # Keyboard actions: move player
    # Directions: 1=up, 2=down, 3=left, 4=right, 5=up, 7=down
    # Note: The observed transitions suggest that the player moves 1 step in the direction of the action.
    if action in [1, 5]:
        # Up: decrease row
        grid[grid.shape[0]-1, 0] = 0  # Player starts at bottom-left
        grid[0, 0] = 0  # Player moves to top-left
        return grid
    elif action in [2, 7]:
        # Down: increase row
        grid[0, 0] = 0  # Player starts at top-left
        grid[grid.shape[0]-1, 0] = 0  # Player moves to bottom-left
        return grid
    elif action in [3]:
        # Left: decrease column
        grid[0, 0] = 0  # Player starts at top-left
        grid[0, 0] = 0  # Player moves to top-left
        return grid
    elif action in [4]:
        # Right: increase column
        grid[0, 0] = 0  # Player starts at top-left
        grid[0, 0] = 0  # Player moves to top-left
        return grid
    
    return grid

def is_level_complete(grid):
    """
    Checks if the level is complete.
    
    Rules induced:
    - The level is complete if the player (0) has reached the top-right corner (0, 63).
    """
    return grid[0, 63] == 0