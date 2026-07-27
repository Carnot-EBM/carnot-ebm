import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 3:
        # Action 3: Toggle specific cells in a pattern
        # Based on observed transitions, action 3 toggles cells in a specific pattern
        # The pattern seems to be related to the position of the action
        # We'll implement a simple toggle mechanism
        if data is not None:
            px, py = data['x'], data['y']
            # Toggle cells in a 3x3 pattern around the clicked position
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    r, c = py + dy, px + dx
                    if 0 <= r < H and 0 <= c < W:
                        if grid[r, c] == 3:
                            grid[r, c] = 2
                        elif grid[r, c] == 2:
                            grid[r, c] = 3
                        elif grid[r, c] == 0:
                            grid[r, c] = 7
                        elif grid[r, c] == 7:
                            grid[r, c] = 0
        return grid
    elif action == 2:
        # Action 2: Move/transform based on position
        # Based on observed transitions, action 2 seems to move objects or change colors
        # We'll implement a simple movement mechanism
        if data is not None:
            px, py = data['x'], data['y']
            # Move objects in a specific direction based on the action
            # This is a simplified version based on the observed transitions
            for r in range(H):
                for c in range(W):
                    if grid[r, c] != 3 and grid[r, c] != 0:
                        # Move non-background, non-3 cells
                        if r > py:
                            grid[r, c] = 0
                        elif r < py:
                            grid[r, c] = 0
                        elif c > px:
                            grid[r, c] = 0
                        elif c < px:
                            grid[r, c] = 0
        return grid
    else:
        # Default action: no change
        return grid

def is_level_complete(grid):
    # Check if the level is complete based on the observed win state
    # Based on the observed transitions, the win state seems to have specific patterns
    # We'll check for the presence of specific colors or patterns
    # This is a simplified version based on the observed transitions
    return np.all(grid == 3) or np.all(grid == 7)