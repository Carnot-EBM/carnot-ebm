import numpy as np

def engine(grid, action, data):
    h, w = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move player left
        # Find player (color 0)
        player_pos = np.argwhere(new_grid == 0)
        if len(player_pos) == 1:
            py, px = player_pos[0]
            if px > 0:
                # Move player left
                new_grid[py, px] = new_grid[py, px - 1]
                new_grid[py, px - 1] = 0
                # Check for collected items (color 10)
                if new_grid[py, px - 1] == 10:
                    new_grid[py, px - 1] = 11  # Collected item becomes 11
                # Check for obstacles (color 12)
                if new_grid[py, px - 1] == 12:
                    # Player cannot move into obstacle, so no change
                    pass
                # Check for walls (color 5)
                if new_grid[py, px - 1] == 5:
                    # Player cannot move into wall, so no change
                    pass
    elif action == 3:
        # Action 3: Move player right
        player_pos = np.argwhere(new_grid == 0)
        if len(player_pos) == 1:
            py, px = player_pos[0]
            if px < w - 1:
                # Move player right
                new_grid[py, px] = new_grid[py, px + 1]
                new_grid[py, px + 1] = 0
                # Check for collected items (color 10)
                if new_grid[py, px + 1] == 10:
                    new_grid[py, px + 1] = 11
                # Check for obstacles (color 12)
                if new_grid[py, px + 1] == 12:
                    pass
                # Check for walls (color 5)
                if new_grid[py, px + 1] == 5:
                    pass
    elif action == 2:
        # Action 2: Move player up
        player_pos = np.argwhere(new_grid == 0)
        if len(player_pos) == 1:
            py, px = player_pos[0]
            if py > 0:
                # Move player up
                new_grid[py, px] = new_grid[py - 1, px]
                new_grid[py - 1, px] = 0
                # Check for collected items (color 10)
                if new_grid[py - 1, px] == 10:
                    new_grid[py - 1, px] = 11
                # Check for obstacles (color 12)
                if new_grid[py - 1, px] == 12:
                    pass
                # Check for walls (color 5)
                if new_grid[py - 1, px] == 5:
                    pass
    elif action == 4:
        # Action 4: Move player down
        player_pos = np.argwhere(new_grid == 0)
        if len(player_pos) == 1:
            py, px = player_pos[0]
            if py < h - 1:
                # Move player down
                new_grid[py, px] = new_grid[py + 1, px]
                new_grid[py + 1, px] = 0
                # Check for collected items (color 10)
                if new_grid[py + 1, px] == 10:
                    new_grid[py + 1, px] = 11
                # Check for obstacles (color 10)
                if new_grid[py + 1, px] == 12:
                    pass
                # Check for walls (color 5)
                if new_grid[py + 1, px] == 5:
                    pass
    
    return new_grid

def is_level_complete(grid):
    # Check if all 10s are collected (converted to 11s)
    # Count 10s and 11s
    count_10 = np.sum(grid == 10)
    count_11 = np.sum(grid == 11)
    # If no 10s remain, level is complete
    return count_10 == 0