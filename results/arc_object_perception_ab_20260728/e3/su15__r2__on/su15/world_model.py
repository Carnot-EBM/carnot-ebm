import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        if grid[py, px] == 0:
            grid[py, px] = 5
            # Check for adjacent 5s to form a 3x3 block
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                        if grid[ny, nx] == 5:
                            neighbors.append((ny, nx))
            if len(neighbors) == 4:
                # Form a 3x3 block of 5s
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                            grid[ny, nx] = 5
            # Check for adjacent 5s to form a 1x3 or 3x1 block
            if grid[py, px] == 5:
                # Horizontal 1x3
                if grid[py, px-1] == 5 and grid[py, px-2] == 5:
                    grid[py, px-1] = 5
                    grid[py, px-2] = 5
                # Vertical 3x1
                if grid[py-1, px] == 5 and grid[py-2, px] == 5:
                    grid[py-1, px] = 5
                    grid[py-2, px] = 5
        return grid
    return grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # The win state has specific patterns of 5s and other colors
    # We check for the presence of 3x3 blocks of 5s and other specific patterns
    # This is a simplified check based on the observed win state
    # Check for 3x3 blocks of 5s
    for y in range(grid.shape[0] - 2):
        for x in range(grid.shape[1] - 2):
            if grid[y:y+3, x:x+3].sum() == 9:
                return True
    return False