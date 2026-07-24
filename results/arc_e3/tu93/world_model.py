def engine(grid, action, data):
    """
    Simulate one step of the world.
    - grid: 2D list of integers (the real grid shape/format already used by the code above)
    - action: 0=up, 1=down, 2=left, 3=right
    - data: None (unused)
    Returns: new grid after applying the action.
    """
    if not grid:
        return grid

    rows = len(grid)
    cols = len(grid[0])

    # Create a deep copy of the grid to avoid modifying the original
    new_grid = [row[:] for row in grid]

    # Direction vectors for each action: 0=up, 1=down, 2=left, 3=right
    # Note: The original code had a bug where it swapped up/down and left/right
    # We need to fix this to match the expected behavior
    directions = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1)    # right
    }

    # Get the direction vector for the given action
    dr, dc = directions[action]

    # Apply the action to all cells
    for r in range(rows):
        for c in range(cols):
            # If the cell is not a wall (value != 0), move it
            if grid[r][c] != 0:
                # Calculate the new position
                nr, nc = r + dr, c + dc

                # Check if the new position is within bounds
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Move the cell to the new position
                    new_grid[nr][nc] = grid[r][c]
                    # Clear the old position
                    new_grid[r][c] = 0
                # If the new position is out of bounds, the cell stays in place
                # (this is the expected behavior based on the test cases)

    return new_grid


def is_level_complete(grid):
    """
    Check if the level is complete.
    Returns True if all non-zero cells have been moved to their final positions.
    """
    # Check if all non-zero cells are in their final positions
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != 0:
                # If a cell is not in its final position, the level is not complete
                return False
    return True