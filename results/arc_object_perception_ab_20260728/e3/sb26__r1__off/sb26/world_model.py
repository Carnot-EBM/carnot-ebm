import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        if logical_y < 53:
            return grid
        if logical_y == 53:
            return grid
        if logical_y == 54:
            return grid
        if logical_y == 55:
            return grid
        if logical_y == 56:
            return grid
        if logical_y == 57:
            return grid
        if logical_y == 58:
            return grid
        if logical_y == 59:
            return grid
        if logical_y == 60:
            return grid
        if logical_y == 61:
            return grid
        if logical_y == 62:
            return grid
        if logical_y == 63:
            return grid
    return grid

def is_level_complete(grid):
    return False