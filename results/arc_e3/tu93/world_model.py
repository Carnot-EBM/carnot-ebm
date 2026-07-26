def engine(grid, action, data):
    """
    Simulate the action on the grid.
    grid: 2D list of [row, col, type, state]
    action: 0=up, 1=down, 2=left, 3=right, 4=rotate
    data: unused
    Returns: new grid
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Map actions to direction vectors (row_delta, col_delta)
    # 0=up, 1=down, 2=left, 3=right, 4=rotate
    action_map = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
        4: (0, 0)
    }

    dr, dc = action_map.get(action, (0, 0))

    # Create a deep copy of the grid
    new_grid = [row[:] for row in grid]

    # Handle rotation (action 4)
    if action == 4:
        # Rotate the grid 90 degrees clockwise
        new_grid = [[grid[rows - 1 - r][c] for r in range(rows)] for c in range(cols)]
        return new_grid

    # Handle movement (actions 0, 1, 2, 3)
    if dr == 0 and dc == 0:
        return new_grid

    # Identify all movable blocks (type != 0)
    movable_blocks = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c][2] != 0:
                movable_blocks.append((r, c, grid[r][c]))

    # Move each block
    for r, c, block in movable_blocks:
        nr, nc = r + dr, c + dc

        # Check bounds
        if 0 <= nr < rows and 0 <= nc < cols:
            # Check if target is empty or same type
            if grid[nr][nc][2] == 0 or grid[nr][nc][2] == block[2]:
                # Move the block
                new_grid[nr][nc] = block[:]
                new_grid[r][c] = [r, c, 0, 0]  # Clear old position
            # If blocked by different type, stay in place
        else:
            # Out of bounds, stay in place
            pass

    return new_grid

def is_level_complete(grid):
    """
    Check if the level is complete.
    Returns True if all movable blocks are in their final positions.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Identify all movable blocks
    movable_blocks = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c][2] != 0:
                movable_blocks.append((r, c, grid[r][c]))

    # Check if all blocks are in their final positions
    # A block is in its final position if it's not at the edge of the grid
    for r, c, block in movable_blocks:
        if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
            return False

    return True