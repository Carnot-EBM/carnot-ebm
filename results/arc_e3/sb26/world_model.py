import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        # Check if the action is valid (within bounds)
        if 0 <= px < grid.shape[1] and 0 <= py < grid.shape[0]:
            # Check if the target cell is 4 (the player)
            if new_grid[py, px] == 4:
                # Move the player to the clicked position
                new_grid[py, px] = 4
                # Check for adjacent cells that are not 4 and change them to 4
                # This is a simplified rule based on the observed transitions
                # The player seems to be able to change adjacent cells to 4
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                            if new_grid[ny, nx] != 4:
                                new_grid[ny, nx] = 4
        return new_grid
    else:
        # For other actions, return the grid unchanged
        return grid

def is_level_complete(grid):
    # Check if the grid is complete based on the observed transitions
    # The level is complete if there are no more changes possible
    # This is a simplified rule based on the observed transitions
    # The level is complete if the grid is full of 4s
    return np.all(grid == 4)