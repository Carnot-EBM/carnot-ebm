import numpy as np

def engine(grid, action, data):
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                y, x = py + dy, px + dx
                if 0 <= y < h and 0 <= x < w:
                    if new_grid[y, x] == 0:
                        new_grid[y, x] = 3
        return new_grid
    elif action == 1:
        h, w = grid.shape
        new_grid = grid.copy()
        for y in range(h):
            for x in range(w):
                if grid[y, x] == 1:
                    if x + 1 < w and new_grid[y, x + 1] == 0:
                        new_grid[y, x + 1] = 1
                        new_grid[y, x] = 0
        return new_grid
    elif action == 2:
        h, w = grid.shape
        new_grid = grid.copy()
        for y in range(h):
            for x in range(w):
                if grid[y, x] == 2:
                    if y + 1 < h and new_grid[y + 1, x] == 0:
                        new_grid[y + 1, x] = 2
                        new_grid[y, x] = 0
        return new_grid
    elif action == 3:
        h, w = grid.shape
        new_grid = grid.copy()
        for y in range(h):
            for x in range(w):
                if grid[y, x] == 3:
                    if y - 1 >= 0 and new_grid[y - 1, x] == 0:
                        new_grid[y - 1, x] = 3
                        new_grid[y, x] = 0
        return new_grid
    elif action == 5:
        h, w = grid.shape
        new_grid = grid.copy()
        for y in range(h):
            for x in range(w):
                if grid[y, x] == 5:
                    if x - 1 >= 0 and new_grid[y, x - 1] == 0:
                        new_grid[y, x - 1] = 5
                        new_grid[y, x] = 0
        return new_grid
    elif action == 6:
        h, w = grid.shape
        if data is None:
            return grid
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                y, x = py + dy, px + dx
                if 0 <= y < h and 0 <= x < w:
                    if new_grid[y, x] == 0:
                        new_grid[y, x] = 3
        return new_grid
    elif action == 7:
        h, w = grid.shape
        new_grid = grid.copy()
        for y in range(h):
            for x in range(w):
                if grid[y, x] == 7:
                    if x + 1 < w and new_grid[y, x + 1] == 0:
                        new_grid[y, x + 1] = 7
                        new_grid[y, x] = 0
        return new_grid
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    if h != 64 or w != 64:
        return False
    for y in range(h):
        row = grid[y, :]
        if np.all(row == 2):
            continue
        if np.all(row == 3):
            continue
        if np.all(row == 4):
            continue
        if np.all(row == 5):
            continue
        if np.all(row == 6):
            continue
        if np.all(row == 7):
            continue
        if np.all(row == 10):
            continue
        if np.all(row == 11):
            continue
        if np.all(row == 0):
            continue
        if np.all(row == 1):
            continue
        if np.all(row == 12):
            continue
        if np.all(row == 13):
            continue
        if np.all(row == 14):
            continue
        if np.all(row == 15):
            continue
        if np.all(row == 16):
            continue
        if np.all(row == 17):
            continue
        if np.all(row == 18):
            continue
        if np.all(row == 19):
            continue
        if np.all(row == 20):
            continue
        if np.all(row == 21):
            continue
        if np.all(row == 22):
            continue
        if np.all(row == 23):
            continue
        if np.all(row == 24):
            continue
        if np.all(row == 25):
            continue
        if np.all(row == 26):
            continue
        if np.all(row == 27):
            continue
        if np.all(row == 28):
            continue
        if np.all(row == 29):
            continue
        if np.all(row == 30):
            continue
        if np.all(row == 31):
            continue
        if np.all(row == 32):
            continue
        if np.all(row == 33):
            continue
        if np.all(row == 34):
            continue
        if np.all(row == 35):
            continue
        if np.all(row == 36):
            continue
        if np.all(row == 37):
            continue
        if np.all(row == 38):
            continue
        if np.all(row == 39):
            continue
        if np.all(row == 40):
            continue
        if np.all(row == 41):
            continue
        if np.all(row == 42):
            continue
        if np.all(row == 43):
            continue
        if np.all(row == 44):
            continue
        if np.all(row == 45):
            continue
        if np.all(row == 46):
            continue
        if np.all(row == 47):
            continue
        if np.all(row == 48):
            continue
        if np.all(row == 49):
            continue
        if np.all(row == 50):
            continue
        if np.all(row == 51):
            continue
        if np.all(row == 52):
            continue
        if np.all(row == 53):
            continue
        if np.all(row == 54):
            continue
        if np.all(row == 55):
            continue
        if np.all(row == 56):
            continue
        if np.all(row == 57):
            continue
        if np.all(row == 58):
            continue
        if np.all(row == 59):
            continue
        if np.all(row == 60):
            continue
        if np.all(row == 61):
            continue
        if np.all(row == 62):
            continue
        if np.all(row == 63):
            continue
        return False
    return True