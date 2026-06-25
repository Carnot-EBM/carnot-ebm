import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        if 0 <= py < 64 and 0 <= px < 64:
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < 64 and 0 <= nx < 64:
                        grid[ny, nx] = 14
    return grid

def is_level_complete(grid):
    return False