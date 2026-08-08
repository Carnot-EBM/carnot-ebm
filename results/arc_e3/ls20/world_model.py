import numpy as np

def engine(grid, action, data):
    """
    The world model for the ARC-AGI-3 game 'ls20'.
    The game involves moving a 5x5 block consisting of 2 rows of color 12 
    and 3 rows of color 9.
    ACTION1: Move UP (y -= 5)
    ACTION3: Move LEFT (x -= 5)
    ACTION4: Move RIGHT (x += 5)
    When the block moves, it leaves a trail of color 3 in its previous position.
    """
    grid = np.array(grid)
    
    # Find the current position of the 5x5 block
    # The block is defined by 2 rows of color 12 and 3 rows of color 9.
    y, x = -1, -1
    for r in range(64 - 4):
        for c in range(64 - 4):
            if (grid[r, c:c+5] == 12).all() and \
               (grid[r+1, c:c+5] == 12).all() and \
               (grid[r+2, c:c+5] == 9).all() and \
               (grid[r+3, c:c+5] == 9).all() and \
               (grid[r+4, c:c+5] == 9).all():
                y, x = r, c
                break
        if y != -1:
            break
            
    if y == -1:
        return grid

    # Determine the new position based on the action
    ny, nx = y, x
    if action == 1: # UP
        ny = y - 5
    elif action == 3: # LEFT
        nx = x - 5
    elif action == 4: # RIGHT
        nx = x + 5
    else:
        return grid

    # Keep the block within grid boundaries
    ny = max(0, min(64 - 5, ny))
    nx = max(0, min(64 - 5, nx))

    # If the position didn't change, return the current grid
    if ny == y and nx == x:
        return grid

    new_grid = grid.copy()
    
    # The block leaves a trail of color 3 in its previous position
    new_grid[y:y+5, x:x+5] = 3
    
    # Place the block in its new position
    new_grid[ny:ny+2, nx:nx+5] = 12
    new_grid[ny+2:ny+5, nx:nx+5] = 9
    
    return new_grid

def is_level_complete(grid):
    """
    The level is complete when the 5x5 block reaches the target position (10, 34).
    """
    grid = np.array(grid)
    # Check if the block is at (10, 34)
    # Block: 2 rows of 12, 3 rows of 9
    y, x = 10, 34
    if y + 5 <= 64 and x + 5 <= 64:
        if (grid[y, x:x+5] == 12).all() and \
           (grid[y+1, x:x+5] == 12).all() and \
           (grid[y+2, x:x+5] == 9).all() and \
           (grid[y+3, x:x+5] == 9).all() and \
           (grid[y+4, x:x+5] == 9).all():
            return True
    return False