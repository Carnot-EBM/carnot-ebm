import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid
    
    px, py = data['x'], data['y']
    h, w = grid.shape
    
    # Create a copy to avoid modifying the original grid
    new_grid = grid.copy()
    
    # Define the transformation based on the observed pattern
    # The pattern shows a diagonal line being drawn from top-left to bottom-right
    # starting from the clicked position and extending to the edges
    # The values alternate between 3 and 0
    
    # Calculate the diagonal line
    # The line starts at (px, py) and goes diagonally
    # Based on the pattern, it seems to be a diagonal line with specific values
    
    # Check if the click is within bounds
    if px < 0 or px >= w or py < 0 or py >= h:
        return new_grid
    
    # Create a mask for the diagonal line
    # The line goes from (px, py) to the opposite corner
    # Based on the pattern, it seems to be a diagonal line
    
    # Determine the direction based on the position
    # If px < py, the line goes down-right
    # If px > py, the line goes up-right
    # If px == py, the line goes down-right
    
    # Based on the observed pattern, the line seems to be:
    # Starting from (px, py) and going to the bottom-right corner
    # The values alternate between 3 and 0
    
    # Calculate the diagonal line
    for i in range(h):
        for j in range(w):
            # Check if the cell is on the diagonal line
            # The line goes from (px, py) to the bottom-right corner
            if i >= py and j >= px:
                # Check if the cell is on the diagonal
                if i - py == j - px:
                    # Determine the value based on the position
                    # Based on the pattern, the values alternate between 3 and 0
                    if (px + py + i + j) % 2 == 0:
                        new_grid[i, j] = 3
                    else:
                        new_grid[i, j] = 0
    
    return new_grid

def is_level_complete(grid):
    # Check if the grid is complete based on the win condition
    # Based on the observed pattern, the win condition is when all cells are filled
    # or when a specific pattern is achieved
    
    # For this game, the win condition seems to be when all cells are filled
    # or when a specific pattern is achieved
    
    # Check if all cells are filled
    if np.all(grid > 0):
        return True
    
    # Check if the grid has a specific pattern
    # Based on the observed pattern, the win condition is when all cells are filled
    return False