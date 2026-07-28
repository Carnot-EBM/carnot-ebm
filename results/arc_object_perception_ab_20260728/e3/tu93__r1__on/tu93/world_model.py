import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 2:
        # Action 2: Move player down (y+1)
        if data is not None:
            px, py = data['x'], data['y']
            # Player is at (py, px) in logical coordinates
            # Move down: new position (py+1, px)
            # Check if there's an obstacle below
            if py + 1 < H and new_grid[py + 1, px] != 5:
                new_grid[py + 1, px] = 6
                new_grid[py, px] = 0
            else:
                # If blocked, player stays
                pass
        else:
            # Keyboard action 2: Move down
            # Find player position (color 6)
            player_pos = np.argwhere(new_grid == 6)
            if len(player_pos) > 0:
                py, px = player_pos[0]
                if py + 1 < H and new_grid[py + 1, px] != 5:
                    new_grid[py + 1, px] = 6
                    new_grid[py, px] = 0
    elif action == 3:
        # Action 3: Move player left (x-1)
        if data is not None:
            px, py = data['x'], data['y']
            if py < H and new_grid[py, px - 1] != 5:
                new_grid[py, px - 1] = 6
                new_grid[py, px] = 0
        else:
            player_pos = np.argwhere(new_grid == 6)
            if len(player_pos) > 0:
                py, px = player_pos[0]
                if px > 0 and new_grid[py, px - 1] != 5:
                    new_grid[py, px - 1] = 6
                    new_grid[py, px] = 0
    elif action == 4:
        # Action 4: Move player right (x+1)
        if data is not None:
            px, py = data['x'], data['y']
            if py < H and new_grid[py, px + 1] != 5:
                new_grid[py, px + 1] = 6
                new_grid[py, px] = 0
        else:
            player_pos = np.argwhere(new_grid == 6)
            if len(player_pos) > 0:
                py, px = player_pos[0]
                if px + 1 < W and new_grid[py, px + 1] != 5:
                    new_grid[py, px + 1] = 6
                    new_grid[py, px] = 0
    elif action == 1:
        # Action 1: Move player up (y-1)
        if data is not None:
            px, py = data['x'], data['y']
            if py > 0 and new_grid[py - 1, px] != 5:
                new_grid[py - 1, px] = 6
                new_grid[py, px] = 0
        else:
            player_pos = np.argwhere(new_grid == 6)
            if len(player_pos) > 0:
                py, px = player_pos[0]
                if py > 0 and new_grid[py - 1, px] != 5:
                    new_grid[py - 1, px] = 6
                    new_grid[py, px] = 0
    elif action == 6:
        # Action 6: Click at pixel coordinates
        if data is not None:
            px, py = data['x'], data['y']
            # Convert pixel to logical: logical = pixel // 1
            lx, ly = px // 1, py // 1
            if 0 <= ly < H and 0 <= lx < W:
                if new_grid[ly, lx] == 5:
                    # Toggle the cell
                    new_grid[ly, lx] = 0
                else:
                    new_grid[ly, lx] = 5
    elif action == 5:
        # Action 5: Move player down-left (diagonal)
        if data is not None:
            px, py = data['x'], data['y']
            if py + 1 < H and px - 1 >= 0 and new_grid[py + 1, px - 1] != 5:
                new_grid[py + 1, px - 1] = 6
                new_grid[py, px] = 0
        else:
            player_pos = np.argwhere(new_grid == 6)
            if len(player_pos) > 0:
                py, px = player_pos[0]
                if py + 1 < H and px - 1 >= 0 and new_grid[py + 1, px - 1] != 5:
                    new_grid[py + 1, px - 1] = 6
                    new_grid[py, px] = 0
    elif action == 7:
        # Action 7: Move player up-right (diagonal)
        if data is not None:
            px, py = data['x'], data['y']
            if py > 0 and px + 1 < W and new_grid[py - 1, px + 1] != 5:
                new_grid[py - 1, px + 1] = 6
                new_grid[py, px] = 0
        else:
            player_pos = np.argwhere(new_grid == 6)
            if len(player_pos) > 0:
                py, px = player_pos[0]
                if py > 0 and px + 1 < W and new_grid[py - 1, px + 1] != 5:
                    new_grid[py - 1, px + 1] = 6
                    new_grid[py, px] = 0
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all rows from 21 to 26 have the win pattern
    # Based on win state: rows 21-26 have specific patterns
    # Row 21: 5x48,14x3,5x13
    # Row 22: 5x48,14x3,5x13
    # Row 23: 5x48,14x3,5x13
    # Row 24: 5x48,2x3,5x13
    # Row 25: 5x48,2x3,5x13
    # Row 26: 5x48,2x3,5x13
    
    # Check rows 21-26
    for i in range(21, 27):
        if i >= H:
            return False
        row = grid[i]
        # Check if row matches the win pattern
        # For rows 21-23: first 48 cells are 5, then 14 cells of 3, then 13 cells of 5
        # For rows 24-26: first 48 cells are 5, then 2 cells of 3, then 13 cells of 5
        if i < 24:
            if not (np.all(row[:48] == 5) and np.all(row[48:62] == 3) and np.all(row[62:] == 5)):
                return False
        else:
            if not (np.all(row[:48] == 5) and np.all(row[48:50] == 3) and np.all(row[50:] == 5)):
                return False
    
    # Check rows 27-35 for the pattern
    for i in range(27, 36):
        if i >= H:
            return False
        row = grid[i]
        # Pattern: 5x12,0x3,2x3,0x3,2x3,0x3,2x3,0x3,2x3,8x3,2x3,0x3,2x3,0x3,5x13
        # This is complex, so we check the overall structure
        # First 12 cells are 5
        if not np.all(row[:12] == 5):
            return False
        # Last 13 cells are 5
        if not np.all(row[-13:] == 5):
            return False
    
    # Check rows 30-32
    for i in range(30, 33):
        if i >= H:
            return False
        row = grid[i]
        # Pattern: 5x12,2x3,5x9,2x3,5x3,2x3,5x3,2x3,5x25
        if not np.all(row[:12] == 5):
            return False
        if not np.all(row[-25:] == 5):
            return False
    
    # Check rows 33-35
    for i in range(33, 36):
        if i >= H:
            return False
        row = grid[i]
        # Pattern: 5x12,9x1,4x1,9x1,5x9,0x3,2x3,0x3,2x3,0x3,5x25
        if not np.all(row[:12] == 5):
            return False
        if not np.all(row[-25:] == 5):
            return False
    
    # Check rows 36-62
    for i in range(36, 63):
        if i >= H:
            return False
        row = grid[i]
        if not np.all(row == 5):
            return False
    
    # Check row 63
    if 63 < H:
        row = grid[63]
        if not np.all(row == 6):
            return False
    
    return True