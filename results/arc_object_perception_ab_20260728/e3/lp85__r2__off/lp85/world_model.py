import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 0:
        # Action 0 is a click at (data['x'], data['y'])
        px, py = data['x'], data['y']
        # Convert pixel to logical
        cx, cy = px // 1, py // 1
        # Check if the clicked cell is a 14 (wall)
        if grid[cy, cx] == 14:
            # Toggle the clicked wall to 3 (empty)
            grid[cy, cx] = 3
            # Apply gravity to the column
            for r in range(H - 1, -1, -1):
                if grid[r, cx] == 3:
                    # Find the next non-empty cell below
                    for nr in range(r + 1, H):
                        if grid[nr, cx] != 3:
                            # Swap
                            grid[r, cx], grid[nr, cx] = grid[nr, cx], grid[r, cx]
                            break
            return grid
    elif action == 1:
        # Action 1 is a move up
        # Move all non-3 cells up
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if grid[r, c] != 3:
                    for nr in range(r - 1, -1, -1):
                        if grid[nr, c] == 3:
                            grid[nr, c] = grid[r, c]
                            grid[r, c] = 3
                            break
        return grid
    elif action == 2:
        # Action 2 is a move down
        # Move all non-3 cells down
        for c in range(W):
            for r in range(H):
                if grid[r, c] != 3:
                    for nr in range(r + 1, H):
                        if grid[nr, c] == 3:
                            grid[nr, c] = grid[r, c]
                            grid[r, c] = 3
                            break
        return grid
    elif action == 3:
        # Action 3 is a move left
        # Move all non-3 cells left
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if grid[r, c] != 3:
                    for nc in range(c - 1, -1, -1):
                        if grid[r, nc] == 3:
                            grid[r, nc] = grid[r, c]
                            grid[r, c] = 3
                            break
        return grid
    elif action == 4:
        # Action 4 is a move right
        # Move all non-3 cells right
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 3:
                    for nc in range(c + 1, W):
                        if grid[r, nc] == 3:
                            grid[r, nc] = grid[r, c]
                            grid[r, c] = 3
                            break
        return grid
    elif action == 5:
        # Action 5 is a move up-left
        # Move all non-3 cells up-left
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if grid[r, c] != 3:
                    for nr in range(r - 1, -1, -1):
                        if grid[nr, c] == 3:
                            grid[nr, c] = grid[r, c]
                            grid[r, c] = 3
                            break
        return grid
    elif action == 6:
        # Action 6 is a move up-right
        # Move all non-3 cells up-right
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if grid[r, c] != 3:
                    for nr in range(r - 1, -1, -1):
                        if grid[nr, c] == 3:
                            grid[nr, c] = grid[r, c]
                            grid[r, c] = 3
                            break
        return grid
    elif action == 7:
        # Action 7 is a move down-left
        # Move all non-3 cells down-left
        for c in range(W):
            for r in range(H):
                if grid[r, c] != 3:
                    for nr in range(r + 1, H):
                        if grid[nr, c] == 3:
                            grid[nr, c] = grid[r, c]
                            grid[r, c] = 3
                            break
        return grid
    return grid

def is_level_complete(grid):
    # Check if the grid is a win state
    # The win state has specific patterns
    # Check if all rows are either all 14 and 3, or have specific patterns
    # Based on the win state, check if the grid matches the win state pattern
    # The win state has rows with 14x1, 3x10, 4x41, 3x12
    # Check if all rows match the win state pattern
    for r in range(grid.shape[0]):
        row = grid[r, :]
        # Check if the row matches the win state pattern
        # The win state has rows with 14x1, 3x10, 4x41, 3x12
        # Check if the row matches the win state pattern
        if not (np.all(row == 14) or np.all(row == 3) or np.all(row == 4) or np.all(row == 15)):
            return False
    return True