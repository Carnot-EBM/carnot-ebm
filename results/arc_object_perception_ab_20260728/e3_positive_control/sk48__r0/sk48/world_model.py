import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 1:
        # Action 1: Move player down (row +1)
        if data is not None:
            px, py = data['x'], data['y']
            grid[py, px] = 1
            grid[py + 1, px] = 0
        else:
            # Move down without data (player at bottom)
            grid[py, px] = 1
            grid[py + 1, px] = 0
    elif action == 2:
        # Action 2: Move player up (row -1)
        if data is not None:
            px, py = data['x'], data['y']
            grid[py, px] = 1
            grid[py - 1, px] = 0
        else:
            grid[py, px] = 1
            grid[py - 1, px] = 0
    elif action == 3:
        # Action 3: Move player right (col +1)
        if data is not None:
            px, py = data['x'], data['y']
            grid[py, px] = 1
            grid[py, px + 1] = 0
        else:
            grid[py, px] = 1
            grid[py, px + 1] = 0
    elif action == 4:
        # Action 4: Move player left (col -1)
        if data is not None:
            px, py = data['x'], data['y']
            grid[py, px] = 1
            grid[py, px - 1] = 0
        else:
            grid[py, px] = 1
            grid[py, px - 1] = 0
    elif action == 5:
        # Action 5: Move player down-right (row +1, col +1)
        if data is not None:
            px, py = data['x'], data['y']
            grid[py, px] = 1
            grid[py + 1, px + 1] = 0
        else:
            grid[py, px] = 1
            grid[py + 1, px + 1] = 0
    elif action == 6:
        # Action 6: Click (no change)
        pass
    elif action == 7:
        # Action 7: Move player up-left (row -1, col -1)
        if data is not None:
            px, py = data['x'], data['y']
            grid[py, px] = 1
            grid[py - 1, px - 1] = 0
        else:
            grid[py, px] = 1
            grid[py - 1, px - 1] = 0
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if the grid matches the win state pattern
    # Win state has specific patterns in rows 6-47
    # Check if rows 6-47 have the pattern: 5x11, 4x42, 5x11
    for i in range(6, 48):
        row = grid[i]
        # Check if row matches the win state pattern
        if not np.array_equal(row, np.array([5]*11 + [4]*42 + [5]*11)):
            return False
    return True