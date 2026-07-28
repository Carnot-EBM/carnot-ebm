import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if py < 11:
            return new_grid
        for r in range(11, min(py, 30)):
            for c in range(11, min(W, 11 + (30 - r) * 2)):
                if grid[r, c] == 5:
                    new_grid[r, c] = 0
        return new_grid

    if action == 2:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if py < 11:
            return new_grid
        for r in range(11, min(py, 30)):
            for c in range(11, min(W, 11 + (30 - r) * 2)):
                if grid[r, c] == 5:
                    new_grid[r, c] = 4
        return new_grid

    if action == 3:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if py < 11:
            return new_grid
        for r in range(11, min(py, 30)):
            for c in range(11, min(W, 11 + (30 - r) * 2)):
                if grid[r, c] == 5:
                    new_grid[r, c] = 4
        return new_grid

    if action == 4:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if py < 11:
            return new_grid
        for r in range(11, min(py, 30)):
            for c in range(11, min(W, 11 + (30 - r) * 2)):
                if grid[r, c] == 5:
                    new_grid[r, c] = 4
        return new_grid

    if action == 5:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if py < 11:
            return new_grid
        for r in range(11, min(py, 30)):
            for c in range(11, min(W, 11 + (30 - r) * 2)):
                if grid[r, c] == 5:
                    new_grid[r, c] = 4
        return new_grid

    if action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if py < 11:
            return new_grid
        for r in range(11, min(py, 30)):
            for c in range(11, min(W, 11 + (30 - r) * 2)):
                if grid[r, c] == 5:
                    new_grid[r, c] = 4
        return new_grid

    if action == 7:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if py < 11:
            return new_grid
        for r in range(11, min(py, 30)):
            for c in range(11, min(W, 11 + (30 - r) * 2)):
                if grid[r, c] == 5:
                    new_grid[r, c] = 4
        return new_grid

    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        row_str = ""
        for c in range(W):
            if grid[r, c] == 5:
                row_str += "5x64"
                break
        if row_str != "5x64":
            return False
    for r in range(H):
        if r < 6:
            continue
        if r >= 6 and r < 11:
            if grid[r, 11] != 4:
                return False
            if grid[r, 11+42] != 5:
                return False
            if grid[r, 11+42+11] != 5:
                return False
        if r >= 11:
            if grid[r, 11] != 4:
                return False
            if grid[r, 11+42] != 5:
                return False
            if grid[r, 11+42+11] != 5:
                return False
    return True