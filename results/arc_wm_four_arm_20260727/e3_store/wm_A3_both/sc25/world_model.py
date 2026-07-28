import numpy as np

def engine(grid, action, data):
    if action == 1:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 0 or py >= grid.shape[0] or px < 0 or px >= grid.shape[1]:
            return grid
        if grid[py, px] == 0:
            return grid
        if grid[py, px] == 15:
            return grid
        if grid[py, px] == 14:
            return grid
        if grid[py, px] == 8:
            return grid
        if grid[py, px] == 4:
            return grid
        if grid[py, px] == 5:
            return grid
        if grid[py, px] == 9:
            return grid
        if grid[py, px] == 13:
            return grid
        return grid
    elif action == 2:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 0 or py >= grid.shape[0] or px < 0 or px >= grid.shape[1]:
            return grid
        if grid[py, px] == 0:
            return grid
        if grid[py, px] == 15:
            return grid
        if grid[py, px] == 14:
            return grid
        if grid[py, px] == 8:
            return grid
        if grid[py, px] == 4:
            return grid
        if grid[py, px] == 5:
            return grid
        if grid[py, px] == 9:
            return grid
        if grid[py, px] == 13:
            return grid
        return grid
    elif action == 3:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 0 or py >= grid.shape[0] or px < 0 or px >= grid.shape[1]:
            return grid
        if grid[py, px] == 0:
            return grid
        if grid[py, px] == 15:
            return grid
        if grid[py, px] == 14:
            return grid
        if grid[py, px] == 8:
            return grid
        if grid[py, px] == 4:
            return grid
        if grid[py, px] == 5:
            return grid
        if grid[py, px] == 9:
            return grid
        if grid[py, px] == 13:
            return grid
        return grid
    elif action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 0 or py >= grid.shape[0] or px < 0 or px >= grid.shape[1]:
            return grid
        if grid[py, px] == 0:
            return grid
        if grid[py, px] == 15:
            return grid
        if grid[py, px] == 14:
            return grid
        if grid[py, px] == 8:
            return grid
        if grid[py, px] == 4:
            return grid
        if grid[py, px] == 5:
            return grid
        if grid[py, px] == 9:
            return grid
        if grid[py, px] == 13:
            return grid
        return grid
    elif action == 5:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 0 or py >= grid.shape[0] or px < 0 or px >= grid.shape[1]:
            return grid
        if grid[py, px] == 0:
            return grid
        if grid[py, px] == 15:
            return grid
        if grid[py, px] == 14:
            return grid
        if grid[py, px] == 8:
            return grid
        if grid[px, py] == 4:
            return grid
        if grid[px, py] == 5:
            return grid
        if grid[px, py] == 9:
            return grid
        if grid[px, py] == 13:
            return grid
        return grid
    elif action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 0 or py >= grid.shape[0] or px < 0 or px >= grid.shape[1]:
            return grid
        if grid[py, px] == 0:
            return grid
        if grid[py, px] == 15:
            return grid
        if grid[py, px] == 14:
            return grid
        if grid[py, px] == 8:
            return grid
        if grid[py, px] == 4:
            return grid
        if grid[py, px] == 5:
            return grid
        if grid[py, px] == 9:
            return grid
        if grid[py, px] == 13:
            return grid
        return grid
    elif action == 7:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 0 or py >= grid.shape[0] or px < 0 or px >= grid.shape[1]:
            return grid
        if grid[py, px] == 0:
            return grid
        if grid[py, px] == 15:
            return grid
        if grid[py, px] == 14:
            return grid
        if grid[py, px] == 8:
            return grid
        if grid[py, px] == 4:
            return grid
        if grid[py, px] == 5:
            return grid
        if grid[py, px] == 9:
            return grid
        if grid[py, px] == 13:
            return grid
        return grid
    return grid

def is_level_complete(grid):
    return False