import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 2:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Action 2: Click at (px, py)
        # Based on observations, this toggles cells in a specific pattern
        # The pattern seems to be related to the position of the click
        # We'll implement a simple toggle mechanism
        # Toggle cells in a 3x3 area around the click position
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = py + dy, px + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if new_grid[ny, nx] == 5:
                        new_grid[ny, nx] = 0
                    else:
                        new_grid[ny, nx] = 5
    elif action == 3:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Action 3: Click at (px, py)
        # Similar to action 2 but with different pattern
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, ny = py + dy, px + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if new_grid[ny, nx] == 5:
                        new_grid[ny, nx] = 0
                    else:
                        new_grid[ny, nx] = 5
    elif action == 4:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Action 4: Click at (px, py)
        # Similar to action 2 but with different pattern
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = py + dy, px + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if new_grid[ny, nx] == 5:
                        new_grid[ny, nx] = 0
                    else:
                        new_grid[ny, nx] = 5
    elif action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Action 6: Click at (px, py)
        # Similar to action 2 but with different pattern
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = py + dy, px + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if new_grid[ny, nx] == 5:
                        new_grid[ny, nx] = 0
                    else:
                        new_grid[ny, nx] = 5
    elif action in [1, 5, 7]:
        # Directional actions
        # Move the player or object
        pass
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all cells are filled with color 5
    # Based on the win state, all cells should be 5
    return np.all(grid == 5)