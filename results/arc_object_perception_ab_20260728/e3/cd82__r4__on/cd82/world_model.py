import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move right
        for r in range(H):
            for c in range(W - 1):
                if grid[r, c] != 0 and grid[r, c + 1] == 0:
                    new_grid[r, c + 1] = grid[r, c]
                    new_grid[r, c] = 0
    elif action == 2:
        # Action 2: Move left
        for r in range(H):
            for c in range(1, W):
                if grid[r, c] != 0 and grid[r, c - 1] == 0:
                    new_grid[r, c - 1] = grid[r, c]
                    new_grid[r, c] = 0
    elif action == 3:
        # Action 3: Move down
        for c in range(W):
            for r in range(H - 1):
                if grid[r, c] != 0 and grid[r + 1, c] == 0:
                    new_grid[r + 1, c] = grid[r, c]
                    new_grid[r, c] = 0
    elif action == 4:
        # Action 4: Move up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] != 0 and grid[r - 1, c] == 0:
                    new_grid[r - 1, c] = grid[r, c]
                    new_grid[r, c] = 0
    elif action == 5:
        # Action 5: Toggle 0 <-> 15
        new_grid = grid.copy()
        new_grid[grid == 0] = 15
        new_grid[grid == 15] = 0
    elif action == 6:
        # Action 6: Click (no-op for this game)
        pass
    elif action == 7:
        # Action 7: Move diagonal (down-right)
        for r in range(H - 1):
            for c in range(W - 1):
                if grid[r, c] != 0 and grid[r + 1, c + 1] == 0:
                    new_grid[r + 1, c + 1] = grid[r, c]
                    new_grid[r, c] = 0
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if the grid matches the win state pattern
    # The win state has specific patterns in the top rows and bottom rows
    # We check for the presence of the win state patterns
    
    # Check top rows (0-17)
    for r in range(18):
        row_str = ','.join(map(str, grid[r]))
        if r < 13:
            # Rows 0-12 have specific patterns
            if r == 0:
                if row_str != '5'*16 + '4'*2 + '3'*46:
                    return False
            elif r == 1:
                if row_str != '5'*16 + '4'*2 + '3'*46:
                    return False
            elif r == 2:
                if row_str != '5'*16 + '4'*2 + '3'*14 + '4'*5 + '3'*1 + '4'*5 + '3'*1 + '4'*5 + '3'*15:
                    return False
            elif r == 3:
                if row_str != '5'*3 + '15'*9 + '12'*1 + '5'*3 + '4'*2 + '3'*14 + '4'*1 + '0'*3 + '4'*1 + '3'*1 + '4'*1 + '15'*3 + '4'*1 + '3'*1 + '4'*1 + '12'*3 + '4'*1 + '3'*15:
                    return False
            elif r == 4:
                if row_str != '5'*3 + '15'*8 + '12'*2 + '5'*3 + '4'*2 + '3'*14 + '4'*1 + '0'*3 + '4'*1 + '3'*1 + '4'*1 + '15'*3 + '4'*1 + '3'*1 + '4'*1 + '12'*3 + '4'*1 + '3'*15:
                    return False
            elif r == 5:
                if row_str != '5'*3 + '15'*7 + '12'*3 + '5'*3 + '4'*2 + '3'*14 + '4'*1 + '0'*3 + '4'*1 + '3'*1 + '4'*1 + '15'*3 + '4'*1 + '3'*1 + '4'*1 + '12'*3 + '4'*1 + '3'*15:
                    return False
            elif r == 6:
                if row_str != '5'*3 + '15'*6 + '12'*4 + '5'*3 + '4'*2 + '3'*14 + '4'*5 + '3'*1 + '4'*5 + '3'*1 + '4'*5 + '3'*15:
                    return False
            elif r == 7:
                if row_str != '5'*3 + '15'*5 + '12'*5 + '5'*3 + '4'*2 + '3'*20 + '0'*5 + '3'*21:
                    return False
            elif r == 8:
                if row_str != '5'*3 + '0'*4 + '12'*6 + '5'*3 + '4'*2 + '3'*46:
                    return False
            elif r == 9:
                if row_str != '5'*3 + '0'*3 + '12'*7 + '5'*3 + '4'*2 + '5'*46:
                    return False
            elif r == 10:
                if row_str != '5'*3 + '0'*2 + '12'*8 + '5'*3 + '4'*2 + '5'*46:
                    return False
            elif r == 11:
                if row_str != '5'*3 + '0'*1 + '12'*9 + '5'*3 + '4'*2 + '5'*46:
                    return False
            elif r == 12:
                if row_str != '5'*3 + '12'*10 + '5'*3 + '4'*2 + '5'*46:
                    return False
            elif r == 13:
                if row_str != '5'*16 + '4'*2 + '5'*46:
                    return False
            elif r == 14:
                if row_str != '5'*16 + '4'*2 + '5'*46:
                    return False
            elif r == 15:
                if row_str != '5'*16 + '4'*2 + '5'*46:
                    return False
            elif r == 16:
                if row_str != '4'*18 + '5'*46:
                    return False
            elif r == 17:
                if row_str != '4'*18 + '5'*46:
                    return False
        elif r == 18:
            if row_str != '5'*64:
                return False
        elif r == 19:
            if row_str != '5'*64:
                return False
        elif r == 20:
            if row_str != '5'*64:
                return False
        elif r == 21:
            if row_str != '5'*64:
                return False
        elif r == 22:
            if row_str != '5'*64:
                return False
        elif r == 23:
            if row_str != '5'*64:
                return False
        elif r == 24:
            if row_str != '5'*25 + '2'*14 + '5'*25:
                return False
        elif r == 25:
            if row_str != '5'*25 + '2'*1 + '15'*12 + '2'*1 + '5'*25:
                return False
        elif r == 26:
            if row_str != '5'*25 + '2'*1 + '15'*12 + '2'*1 + '5'*25:
                return False
        elif r == 27:
            if row_str != '5'*25 + '2'*1 + '15'*12 + '2'*1 + '5'*25:
                return False
        elif r == 28:
            if row_str != '5'*25 + '2'*1 + '15'*12 + '2'*1 + '5'*25:
                return False
        elif r == 29:
            if row_str != '5'*25 + '2'*1 + '15'*12 + '2'*1 + '5'*25:
                return False
        elif r == 30:
            if row_str != '5'*25 + '2'*1 + '15'*12 + '2'*1 + '5'*25:
                return False
        elif r == 31:
            if row_str != '5'*25 + '2'*1 + '15'*12 + '2'*1 + '5'*25:
                return False
        elif r == 32:
            if row_str != '5'*25 + '2'*1 + '5'*12 + '2'*1 + '5'*25:
                return False
        elif r == 33:
            if row_str != '5'*64:
                return False
        elif r == 34:
            if row_str != '5'*27 + '0'*10 + '5'*27:
                return False
        elif r == 35:
            if row_str != '5'*27 + '0'*10 + '5'*27:
                return False
        elif r == 36:
            if row_str != '5'*27 + '0'*10 + '5'*27:
                return False
        elif r == 37:
            if row_str != '5'*27 + '0'*10 + '5'*27:
                return False
        elif r == 38:
            if row_str != '5'*27 + '0'*10 + '5'*27:
                return False
        elif r == 39:
            if row_str != '5'*27 + '0'*10 + '5'*27:
                return False
        elif r == 40:
            if row_str != '5'*27 + '0'*10 + '5'*27:
                return False
        elif r == 41:
            if row_str != '5'*27 + '0'*10 + '5'*27:
                return False
        elif r == 42:
            if row_str != '5'*27 + '0'*10 + '5'*27:
                return False
        elif r == 43:
            if row_str != '5'*27 + '0'*10 + '5'*27:
                return False
        elif r == 44:
            if row_str != '5'*64:
                return False
        elif r == 45:
            if row_str != '5'*64:
                return False
        elif r == 46:
            if row_str != '5'*64:
                return False
        elif r == 47:
            if row_str != '5'*64:
                return False
        elif r == 48:
            if row_str != '5'*64:
                return False
        elif r == 49:
            if row_str != '5'*64:
                return False
        elif r == 50:
            if row_str != '5'*64:
                return False
        elif r == 51:
            if row_str != '5'*64:
                return False
        elif r == 52:
            if row_str != '5'*64:
                return False
        elif r == 53:
            if row_str != '5'*64:
                return False
        elif r == 54:
            if row_str != '5'*64:
                return False
        elif r == 55:
            if row_str != '5'*64:
                return False
        elif r == 56:
            if row_str != '5'*64:
                return False
        elif r == 57:
            if row_str != '5'*64:
                return False
        elif r == 58:
            if row_str != '5'*64:
                return False
        elif r == 59:
            if row_str != '5'*64:
                return False
        elif r == 60:
            if row_str != '5'*64:
                return False
        elif r == 61:
            if row_str != '5'*64:
                return False
        elif r == 62:
            if row_str != '5'*64:
                return False
        elif r == 63:
            if row_str != '4'*64:
                return False
    
    return True