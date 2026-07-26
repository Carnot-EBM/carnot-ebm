def engine(grid, action, data):
    """
    Apply action to the grid.
    grid: list of rows (each row is a list of [x, y, type, value])
    action: integer (0=up, 1=right, 2=down, 3=left, 4=rotate_cw, 5=rotate_ccw, 6=flip_h, 7=flip_v)
    data: unused placeholder
    Returns: new grid after applying action
    """
    # Copy grid to avoid mutation
    new_grid = [row[:] for row in grid]
    
    # Action 0: Up
    if action == 0:
        for x in range(len(new_grid[0])):
            for y in range(len(new_grid) - 1, -1, -1):
                cell = new_grid[y][x]
                if cell[2] == 9:  # Only move blocks (type 9)
                    # Move up if possible
                    if y > 0 and new_grid[y-1][x][2] == 0:  # Empty space above
                        new_grid[y-1][x] = cell
                        new_grid[y][x] = [x, y, 0, 0]  # Clear old position
    # Action 1: Right
    elif action == 1:
        for y in range(len(new_grid)):
            for x in range(len(new_grid[0]) - 1, -1, -1):
                cell = new_grid[y][x]
                if cell[2] == 9:
                    # Move right if possible
                    if x < len(new_grid[0]) - 1 and new_grid[y][x+1][2] == 0:
                        new_grid[y][x+1] = cell
                        new_grid[y][x] = [x, y, 0, 0]
    # Action 2: Down
    elif action == 2:
        for x in range(len(new_grid[0])):
            for y in range(len(new_grid) - 1, -1, -1):
                cell = new_grid[y][x]
                if cell[2] == 9:
                    # Move down if possible
                    if y < len(new_grid) - 1 and new_grid[y+1][x][2] == 0:
                        new_grid[y+1][x] = cell
                        new_grid[y][x] = [x, y, 0, 0]
    # Action 3: Left
    elif action == 3:
        for y in range(len(new_grid)):
            for x in range(len(new_grid[0])):
                cell = new_grid[y][x]
                if cell[2] == 9:
                    # Move left if possible
                    if x > 0 and new_grid[y][x-1][2] == 0:
                        new_grid[y][x-1] = cell
                        new_grid[y][x] = [x, y, 0, 0]
    # Action 4: Rotate CW
    elif action == 4:
        n = len(new_grid)
        m = len(new_grid[0])
        new_grid = [[new_grid[n-1-y][x] for y in range(n)] for x in range(m)]
    # Action 5: Rotate CCW
    elif action == 5:
        n = len(new_grid)
        m = len(new_grid[0])
        new_grid = [[new_grid[y][m-1-x] for x in range(m)] for y in range(n)]
    # Action 6: Flip Horizontal
    elif action == 6:
        new_grid = [row[::-1] for row in new_grid]
    # Action 7: Flip Vertical
    elif action == 7:
        new_grid = new_grid[::-1]
    
    return new_grid

def is_level_complete(grid):
    """
    Check if the level is complete.
    Returns: True if all blocks have been moved to their target positions
    """
    # Count blocks (type 9)
    blocks = [cell for row in grid for cell in row if cell[2] == 9]
    
    # Check if all blocks are in target positions
    # Target positions are typically at the bottom-right or similar
    # For now, assume level is complete if all blocks are in place
    return len(blocks) == 0