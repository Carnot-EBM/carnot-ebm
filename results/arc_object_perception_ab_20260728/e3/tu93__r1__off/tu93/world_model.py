import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 2:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Action 2 is a click that toggles cells in a specific pattern
        # Based on observed transitions, it affects cells around the click position
        # The pattern seems to be a 3x3 area with specific values
        # We'll implement a simple toggle mechanism
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = py + dy, px + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if new_grid[ny, nx] == 5:
                        new_grid[ny, nx] = 9
                    elif new_grid[ny, nx] == 9:
                        new_grid[ny, nx] = 5
                    elif new_grid[ny, nx] == 0:
                        new_grid[ny, nx] = 14
                    elif new_grid[ny, nx] == 14:
                        new_grid[ny, nx] = 0
    
    elif action == 3:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Action 3 is a click that toggles cells in a specific pattern
        # Based on observed transitions, it affects cells around the click position
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = py + dy, px + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if new_grid[ny, nx] == 5:
                        new_grid[ny, nx] = 9
                    elif new_grid[ny, nx] == 9:
                        new_grid[ny, nx] = 5
                    elif new_grid[ny, nx] == 0:
                        new_grid[ny, nx] = 14
                    elif new_grid[ny, nx] == 14:
                        new_grid[ny, nx] = 0
    
    elif action == 4:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Action 4 is a click that toggles cells in a specific pattern
        # Based on observed transitions, it affects cells around the click position
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = py + dy, px + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if new_grid[ny, nx] == 5:
                        new_grid[ny, nx] = 9
                    elif new_grid[ny, nx] == 9:
                        new_grid[ny, nx] = 5
                    elif new_grid[ny, nx] == 0:
                        new_grid[ny, nx] = 14
                    elif new_grid[ny, nx] == 14:
                        new_grid[ny, nx] = 0
    
    return new_grid

def is_level_complete(grid):
    # Check if all cells are filled with color 5
    return np.all(grid == 5)