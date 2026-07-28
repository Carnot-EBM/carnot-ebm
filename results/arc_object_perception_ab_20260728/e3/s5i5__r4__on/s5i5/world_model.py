import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if not (0 <= py < grid.shape[0] and 0 <= px < grid.shape[1]):
            return grid
        if grid[py, px] == 5:
            return grid
        
        target_color = 14 if grid[py, px] == 2 else 11
        if grid[py, px] == 2:
            target_color = 11
        elif grid[py, px] == 14:
            target_color = 2
        elif grid[py, px] == 11:
            target_color = 14
        elif grid[py, px] == 3:
            target_color = 15
        elif grid[py, px] == 15:
            target_color = 3
        
        new_grid = grid.copy()
        new_grid[py, px] = target_color
        
        if target_color == 15:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                        if grid[ny, nx] == 3:
                            new_grid[ny, nx] = 15
        elif target_color == 3:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                        if grid[ny, nx] == 15:
                            new_grid[ny, nx] = 3
        elif target_color == 14:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                        if grid[ny, nx] == 2:
                            new_grid[ny, nx] = 14
        elif target_color == 2:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                        if grid[ny, nx] == 14:
                            new_grid[ny, nx] = 2
        elif target_color == 11:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                        if grid[ny, nx] == 4:
                            new_grid[ny, nx] = 11
        elif target_color == 4:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                        if grid[ny, nx] == 11:
                            new_grid[ny, nx] = 4
        return new_grid
    return grid

def is_level_complete(grid):
    return np.all(grid == 5) or np.all(grid == 15)

def is_level_complete(grid):
    import numpy as np
    grid = np.array(grid)
    if grid.shape[0] != 5 or grid.shape[1] != 5:
        return False
    return np.all(grid == 0)
