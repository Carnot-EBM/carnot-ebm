import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        if grid[py, px] == 0:
            grid[py, px] = 15
            # Check for adjacent 5s to trigger collection
            neighbors = [
                (py - 1, px), (py + 1, px),
                (py, px - 1), (py, px + 1)
            ]
            for ny, nx in neighbors:
                if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                    if grid[ny, nx] == 5:
                        grid[ny, nx] = 0
                        grid[py, px] = 5
            return grid
        return grid
    return grid

def is_level_complete(grid):
    return np.all(grid == 0)