import numpy as np


def engine(grid, action, data):
    grid = grid.copy()
    if action == 6 and data is not None:
        px, py = data['x'], data['y']
        x, y = px // 1, py // 1
        if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
            target_color = grid[y][x]
            # Find matching colored blocks in bottom area (rows 57-60)
            for row in range(57, 61):
                for col in range(18, 46):
                    if grid[row][col] == target_color and target_color != 4:
                        # Toggle: if currently colored, set to 0; if 0, restore color
                        if grid[row][col] != 0:
                            # Clear the 4x4 block at this position
                            for dr in range(4):
                                for dc in range(4):
                                    r, c = row + dr, col + dc
                                    if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                                        grid[r][c] = 0
                        else:
                            # Restore the 4x4 block
                            for dr in range(4):
                                for dc in range(4):
                                    r, c = row + dr, col + dc
                                    if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                                        grid[r][c] = target_color
    elif action == 5:
        # Move player/indicator right by 1 pixel on row 53
        if 0 <= 62 < grid.shape[1]:
            old_val = grid[53][63]
            grid[53][63] = 3
            if 0 <= 62 < grid.shape[1]:
                grid[53][62] = old_val
    return grid


def is_level_complete(grid):
    # Check if all bottom blocks are cleared (set to 0)
    for row in range(57, 61):
        for col in range(18, 46):
            if grid[row][col] != 4 and grid[row][col] != 0:
                return False
    return True