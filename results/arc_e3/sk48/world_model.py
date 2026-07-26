import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 1:
        # Action 1: Move left
        return move(grid, 0)
    elif action == 2:
        # Action 2: Move right
        return move(grid, 1)
    elif action == 3:
        # Action 3: Move up
        return move(grid, -1)
    elif action == 4:
        # Action 4: Move down
        return move(grid, 1)
    elif action == 5:
        # Action 5: Toggle color 5
        return toggle(grid, 5)
    elif action == 6:
        # Action 6: Click at pixel data
        if data is not None:
            px, py = data['x'], data['y']
            return click(grid, px, py)
        return grid.copy()
    elif action == 7:
        # Action 7: Toggle color 4
        return toggle(grid, 4)
    return grid.copy()

def move(grid, direction):
    H, W = grid.shape
    new_grid = grid.copy()
    # Identify non-background cells
    mask = grid != 0
    if np.sum(mask) == 0:
        return new_grid
    # Determine direction vector
    if direction == 0:  # Left
        dy, dx = 0, -1
    elif direction == 1:  # Right
        dy, dx = 0, 1
    elif direction == -1:  # Up
        dy, dx = -1, 0
    elif direction == 1:  # Down (duplicate check)
        dy, dx = 1, 0
    else:
        return new_grid
    
    # Move all non-zero cells in direction
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 0:
                # Find new position
                new_r, new_c = r, c
                while True:
                    nr, nc = new_r + dy, new_c + dx
                    if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                        new_r, new_c = nr, nc
                    else:
                        break
                # If moved, update grid
                if new_r != r or new_c != c:
                    new_grid[r, c] = 0
                    new_grid[new_r, new_c] = grid[r, c]
    return new_grid

def toggle(grid, color):
    H, W = grid.shape
    new_grid = grid.copy()
    # Toggle color: change color to 0 if it's the target color, else keep
    # Based on observations, action 3 and 7 toggle specific colors
    mask = grid == color
    new_grid[mask] = 0
    return new_grid

def click(grid, px, py):
    H, W = grid.shape
    new_grid = grid.copy()
    # Click action does not change grid based on observations
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all cells are filled with the same color
    unique_colors = np.unique(grid)
    return len(unique_colors) == 1 and unique_colors[0] != 0