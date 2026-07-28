import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if r == py and c == px:
                    new_grid[r, c] = 1
                elif r == py and c == px + 1:
                    new_grid[r, c] = 1
                elif r == py and c == px - 1:
                    new_grid[r, c] = 1
                elif r == py + 1 and c == px:
                    new_grid[r, c] = 1
                elif r == py - 1 and c == px:
                    new_grid[r, c] = 1
                elif r == py and c == px + 2:
                    new_grid[r, c] = 1
                elif r == py and c == px - 2:
                    new_grid[r, c] = 1
                elif r == py + 1 and c == px + 1:
                    new_grid[r, c] = 1
                elif r == py - 1 and c == px + 1:
                    new_grid[r, c] = 1
                elif r == py + 1 and c == px - 1:
                    new_grid[r, c] = 1
                elif r == py - 1 and c == px - 1:
                    new_grid[r, c] = 1
                elif r == py + 1 and c == px + 2:
                    new_grid[r, c] = 1
                elif r == py + 1 and c == px - 2:
                    new_grid[r, c] = 1
                elif r == py - 1 and c == px + 2:
                    new_grid[r, c] = 1
                elif r == py - 1 and c == px - 2:
                    new_grid[r, c] = 1
                elif r == py + 2 and c == px:
                    new_grid[r, c] = 1
                elif r == py - 2 and c == px:
                    new_grid[r, c] = 1
                elif r == py + 2 and c == px + 1:
                    new_grid[r, c] = 1
                elif r == py + 2 and c == px - 1:
                    new_grid[r, c] = 1
                elif r == py + 2 and c == px + 2:
                    new_grid[r, c] = 1
                elif r == py + 2 and c == px - 2:
                    new_grid[r, c] = 1
                elif r == py - 2 and c == px + 1:
                    new_grid[r, c] = 1
                elif r == py - 2 and c == px - 1:
                    new_grid[r, c] = 1
                elif r == py - 2 and c == px + 2:
                    new_grid[r, c] = 1
                elif r == py - 2 and c == px - 2:
                    new_grid[r, c] = 1
        return new_grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for r in range(H):
        row_str = ""
        for c in range(W):
            row_str += str(grid[r, c])
        if row_str != "0" * 64:
            return False
    return True