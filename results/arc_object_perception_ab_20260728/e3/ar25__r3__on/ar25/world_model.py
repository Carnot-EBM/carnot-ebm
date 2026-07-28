import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 3:
        # Action 3: Click at data['x'], data['y'] (pixel coords)
        # This action toggles a vertical line of cells in the clicked column
        # The line starts from the clicked row and goes down to the bottom
        # The cells toggled are those with color 9 (or 0) and become 5 (or 0)
        # Based on the observed transitions, this action toggles cells in a specific pattern
        # The pattern is: toggle cells in the clicked column from the clicked row down to the bottom
        # The cells that are toggled are those that are part of the "wall" structure (color 9 or 0)
        # and become color 5 (or 0)
        
        # Convert pixel coords to logical coords
        px, py = data['x'], data['y']
        col = px // 1
        row = py // 1
        
        # Toggle cells in the clicked column from the clicked row down to the bottom
        # The cells that are toggled are those that are part of the "wall" structure (color 9 or 0)
        # and become color 5 (or 0)
        # Based on the observed transitions, the action toggles cells in a specific pattern
        # The pattern is: toggle cells in the clicked column from the clicked row down to the bottom
        # The cells that are toggled are those that are part of the "wall" structure (color 9 or 0)
        # and become color 5 (or 0)
        
        # Create a copy of the grid
        new_grid = grid.copy()
        
        # Toggle cells in the clicked column from the clicked row down to the bottom
        # The cells that are toggled are those that are part of the "wall" structure (color 9 or 0)
        # and become color 5 (or 0)
        for r in range(row, H):
            if new_grid[r, col] == 9 or new_grid[r, col] == 0:
                new_grid[r, col] = 5
            else:
                new_grid[r, col] = 0
        
        return new_grid
    elif action == 2:
        # Action 2: Click at data['x'], data['y'] (pixel coords)
        # This action toggles a horizontal line of cells in the clicked row
        # The line starts from the clicked column and goes to the right
        # The cells toggled are those with color 9 (or 0) and become 5 (or 0)
        
        # Convert pixel coords to logical coords
        px, py = data['x'], data['y']
        col = px // 1
        row = py // 1
        
        # Toggle cells in the clicked row from the clicked column to the right
        # The cells that are toggled are those that are part of the "wall" structure (color 9 or 0)
        # and become color 5 (or 0)
        new_grid = grid.copy()
        
        for c in range(col, W):
            if new_grid[row, c] == 9 or new_grid[row, c] == 0:
                new_grid[row, c] = 5
            else:
                new_grid[row, c] = 0
        
        return new_grid
    else:
        # For other actions, return the grid unchanged
        return grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on the observed win state, the win state has a specific pattern
    # The win state has a vertical line of color 0 in the middle of the grid
    # and a horizontal line of color 0 in the middle of the grid
    # The win state also has a specific pattern of color 9 and 11
    
    H, W = grid.shape
    
    # Check for the vertical line of color 0 in the middle of the grid
    mid_col = W // 2
    for r in range(H):
        if grid[r, mid_col] != 0:
            return False
    
    # Check for the horizontal line of color 0 in the middle of the grid
    mid_row = H // 2
    for c in range(W):
        if grid[mid_row, c] != 0:
            return False
    
    # Check for the specific pattern of color 9 and 11
    # The win state has a specific pattern of color 9 and 11
    # This pattern is: a vertical line of color 9 on the left side of the grid
    # and a vertical line of color 11 on the right side of the grid
    
    # Check for the vertical line of color 9 on the left side of the grid
    for r in range(H):
        if grid[r, 0] != 9:
            return False
    
    # Check for the vertical line of color 11 on the right side of the grid
    for r in range(H):
        if grid[r, W-1] != 11:
            return False
    
    return True