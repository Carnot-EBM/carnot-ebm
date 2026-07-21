import numpy as np

def engine(grid, action, data):
    """
    Executes an action on the grid based on the provided action ID and data.
    Refactored to use general rules and avoid NumPy array ambiguity errors.
    """
    grid = np.array(grid)
    rows, cols = grid.shape

    # Movement rules for actions 0-3 (Up, Down, Left, Right)
    if 0 <= action <= 3:
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves[action]
        # Find the first non-zero object to move
        coords = np.argwhere(grid != 0)
        if coords.size > 0:
            r, c = coords[0]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                grid[nr, nc] = grid[r, c]
                grid[r, c] = 0
        return grid.tolist()

    # Color modification rules for actions 4-5
    if action == 4: # Change object color
        coords = np.argwhere(grid != 0)
        if coords.size > 0:
            grid[coords[0][0], coords[0][1]] = data.get('color', 1)
        return grid.tolist()
    
    if action == 5: # Fill entire grid
        grid[:] = data.get('color', 1)
        return grid.tolist()

    # Special rule for action 6 (Fixing the ValueError)
    if action == 6:
        # Use np.array_equal to avoid "truth value of an array is ambiguous" error
        target = np.array(data.get('target', []))
        if target.size > 0 and np.array_equal(grid, target):
            grid[:] = 0
        return grid.tolist()

    return grid.tolist()

def is_level_complete(grid):
    """
    Determines if the level is complete. 
    A level is typically complete if the grid is uniform in color.
    """
    grid = np.array(grid)
    if grid.size == 0:
        return True
    # Check if all elements are the same as the first element
    return np.all(grid == grid[0, 0])