import numpy as np

def engine(grid, action, data):
    h, w = grid.shape
    
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cy, cx = py + dy, px + dx
                if 0 <= cy < h and 0 <= cx < w:
                    if grid[cy, cx] == 3:
                        grid[cy, cx] = 0
    elif action == 1:
        for y in range(h):
            for x in range(w):
                if grid[y, x] == 7:
                    grid[y, x] = 5
    elif action == 2:
        for y in range(h):
            for x in range(w):
                if grid[y, x] == 5:
                    grid[y, x] = 7
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    if grid[h-1, w-1] != 4:
        return False
    for y in range(h):
        for x in range(w):
            if grid[y, x] == 2:
                return False
    return True