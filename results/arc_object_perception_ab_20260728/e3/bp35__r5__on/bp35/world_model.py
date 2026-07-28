import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if grid[py, px] != 5:
            grid = grid.copy()
            grid[py, px] = 15
            grid[py, px + 1] = 15
            grid[py + 1, px] = 15
            grid[py + 1, px + 1] = 15
    else:
        if action == 1:
            dir = 0
        elif action == 2:
            dir = 1
        elif action == 3:
            dir = 2
        elif action == 4:
            dir = 3
        elif action == 5:
            dir = 4
        elif action == 7:
            dir = 5
        else:
            return grid
        
        grid = grid.copy()
        if dir == 0:
            grid[37, 25:38] = 10
            grid[38, 25:38] = 10
            grid[39, 25:38] = 10
            grid[40, 25:38] = 10
            grid[41, 25:38] = 10
            grid[63, 7] = 15
        elif dir == 1:
            grid[37, 31:38] = 10
            grid[38, 31:38] = 10
            grid[39, 31:38] = 10
            grid[40, 31:38] = 10
            grid[41, 31:38] = 10
            grid[63, 8] = 15
        elif dir == 2:
            grid[37, 37:44] = 10
            grid[38, 37:44] = 10
            grid[39, 37:44] = 10
            grid[40, 37:44] = 10
            grid[41, 37:44] = 10
            grid[63, 9] = 15
        elif dir == 3:
            grid[37, 19:26] = 10
            grid[38, 19:26] = 10
            grid[39, 19:26] = 10
            grid[40, 19:26] = 10
            grid[41, 19:26] = 10
            grid[63, 10] = 15
        elif dir == 4:
            grid[37, 25:38] = 10
            grid[38, 25:38] = 10
            grid[39, 25:38] = 10
            grid[40, 25:38] = 10
            grid[41, 25:38] = 10
            grid[63, 11] = 15
        elif dir == 5:
            grid[37, 31:38] = 10
            grid[38, 31:38] = 10
            grid[39, 31:38] = 10
            grid[40, 31:38] = 10
            grid[41, 31:38] = 10
            grid[63, 12] = 15
            
    return grid

def is_level_complete(grid):
    return np.all(grid == 5) or np.all(grid == 10) or np.all(grid == 14) or np.all(grid == 15) or np.all(grid == 3) or np.all(grid == 0)