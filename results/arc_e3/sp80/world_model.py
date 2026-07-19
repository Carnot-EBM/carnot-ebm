def engine(grid, action, data):
    """
    Simulates one step of the game.
    
    Args:
        grid: 2D numpy array of the current game state.
        action: Integer action code (0=up, 1=down, 2=left, 3=right).
        data: Dictionary containing game metadata (e.g., level, score).
    
    Returns:
        Updated grid after applying the action.
    """
    import numpy as np
    
    # Convert action to direction vector
    directions = {
        0: np.array([-1, 0]),  # Up
        1: np.array([1, 0]),   # Down
        2: np.array([0, -1]),  # Left
        3: np.array([0, 1])    # Right
    }
    
    if action not in directions:
        return grid
    
    direction = directions[action]
    rows, cols = grid.shape
    
    # Create a copy of the grid to store the result
    new_grid = grid.copy()
    
    # Helper function to process a single row or column
    def process_line(line):
        # Remove zeros
        line = line[line != 0]
        
        # Merge adjacent equal values
        i = 0
        while i < len(line) - 1:
            if line[i] == line[i+1]:
                line[i] *= 2
                line[i+1] = 0
                i += 2
            else:
                i += 1
        
        # Remove zeros again after merge
        line = line[line != 0]
        
        # Pad with zeros to original length
        line = np.pad(line, (0, len(line) - len(line)), mode='constant')
        return line

    # Process based on direction
    if direction[0] == -1:  # Up
        for c in range(cols):
            col = new_grid[:, c]
            new_grid[:, c] = process_line(col)
            
    elif direction[0] == 1:  # Down
        for c in range(cols):
            col = new_grid[:, c][::-1]
            new_grid[:, c] = process_line(col)[::-1]
            
    elif direction[1] == -1:  # Left
        for r in range(rows):
            row = new_grid[r, :]
            new_grid[r, :] = process_line(row)
            
    elif direction[1] == 1:  # Right
        for r in range(rows):
            row = new_grid[r, :][::-1]
            new_grid[r, :] = process_line(row)[::-1]
    
    return new_grid

def is_level_complete(grid):
    """
    Checks if the current level is complete.
    
    Args:
        grid: 2D numpy array of the current game state.
    
    Returns:
        Boolean indicating if the level is complete.
    """
    import numpy as np
    
    rows, cols = grid.shape
    
    # Check if there are any empty cells
    if np.any(grid == 0):
        return False
    
    # Check if there are any adjacent cells with the same value
    for r in range(rows):
        for c in range(cols):
            # Check right neighbor
            if c + 1 < cols and grid[r, c] == grid[r, c + 1]:
                return False
            # Check bottom neighbor
            if r + 1 < rows and grid[r, c] == grid[r + 1, c]:
                return False
    
    # If no empty cells and no adjacent equal values, level is complete
    return True