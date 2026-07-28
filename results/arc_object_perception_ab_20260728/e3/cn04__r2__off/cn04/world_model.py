import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 4:
        if data is None:
            return new_grid
        
        # Parse delta runs
        runs = []
        for line in data.split('\n'):
            if not line.strip():
                continue
            parts = line.strip().split('c')
            row = int(parts[0][1:])
            cols = parts[1].split(':')
            col_start = int(cols[0][1:])
            run_info = []
            for col_part in cols[1:]:
                val_part = col_part.split('x')
                val = int(val_part[0])
                count = int(val_part[1])
                run_info.append((val, count))
            runs.append((row, col_start, run_info))
        
        # Apply changes
        for row, col_start, run_info in runs:
            col = col_start
            for val, count in run_info:
                new_grid[row, col:col+count] = val
                col += count
                
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    
    # Check if all cells are 12 or 4
    unique_vals = np.unique(grid)
    if not (np.all((unique_vals == 12) | (unique_vals == 4))):
        return False
    
    # Check specific pattern for win state
    # Top rows should be 12, 4, 12
    if not np.all(grid[0, 0:16] == 12) or not np.all(grid[0, 16:48] == 4) or not np.all(grid[0, 48:64] == 12):
        return False
    
    # Check bottom rows
    if not np.all(grid[63, :] == 12):
        return False
    
    # Check middle rows for specific pattern
    # Rows 11-13 should have 12, 0, 12
    if not np.all(grid[11, 0:11] == 12) or not np.all(grid[11, 11:20] == 0) or not np.all(grid[11, 20:64] == 12):
        return False
    
    # Check rows 14-16
    if not np.all(grid[14, 0:11] == 12) or not np.all(grid[14, 11:14] == 0) or not np.all(grid[14, 14:17] == 12) or not np.all(grid[14, 17:20] == 0) or not np.all(grid[14, 20:44] == 12) or not np.all(grid[14, 44:52] == 8) or not np.all(grid[14, 52:56] == 14) or not np.all(grid[14, 56:64] == 12):
        return False
    
    # Check rows 17-19
    if not np.all(grid[17, 0:11] == 12) or not np.all(grid[17, 11:14] == 0) or not np.all(grid[17, 14:17] == 12) or not np.all(grid[17, 17:20] == 0) or not np.all(grid[17, 20:47] == 12) or not np.all(grid[17, 47:61] == 14) or not np.all(grid[17, 61:64] == 12):
        return False
    
    # Check rows 20-22
    if not np.all(grid[20, 0:11] == 12) or not np.all(grid[20, 11:14] == 0) or not np.all(grid[20, 14:17] == 12) or not np.all(grid[20, 17:20] == 0) or not np.all(grid[20, 20:23] == 8) or not np.all(grid[20, 23:47] == 12) or not np.all(grid[20, 47:50] == 14) or not np.all(grid[20, 50:53] == 8) or not np.all(grid[20, 53:64] == 12):
        return False
    
    # Check rows 23-25
    if not np.all(grid[23, 0:11] == 12) or not np.all(grid[23, 11:44] == 0) or not np.all(grid[23, 44:58] == 14) or not np.all(grid[23, 58:64] == 12):
        return False
    
    # Check rows 26-28
    if not np.all(grid[26, 0:11] == 12) or not np.all(grid[26, 11:14] == 0) or not np.all(grid[26, 14:17] == 8) or not np.all(grid[26, 17:47] == 12) or not np.all(grid[26, 47:50] == 14) or not np.all(grid[26, 50:53] == 8) or not np.all(grid[26, 53:64] == 12):
        return False
    
    # Check rows 29-31
    if not np.all(grid[29, 0:38] == 12) or not np.all(grid[29, 38:46] == 8) or not np.all(grid[29, 46:52] == 12) or not np.all(grid[29, 52:56] == 14) or not np.all(grid[29, 56:64] == 12):
        return False
    
    # Check rows 32-34
    if not np.all(grid[32, 0:38] == 12) or not np.all(grid[32, 38:52] == 14) or not np.all(grid[32, 52:64] == 12):
        return False
    
    # Check rows 35-37
    if not np.all(grid[35, 0:23] == 12) or not np.all(grid[35, 23:34] == 11) or not np.all(grid[35, 34:64] == 12):
        return False
    
    # Check rows 38-40
    if not np.all(grid[38, 0:17] == 12) or not np.all(grid[38, 17:25] == 8) or not np.all(grid[38, 25:28] == 12) or not np.all(grid[38, 28:31] == 11) or not np.all(grid[38, 31:34] == 12) or not np.all(grid[38, 34:37] == 11) or not np.all(grid[38, 37:45] == 8) or not np.all(grid[38, 45:64] == 12):
        return False
    
    # Check rows 41-43
    if not np.all(grid[41, 0:17] == 12) or not np.all(grid[41, 17:28] == 11) or not np.all(grid[41, 28:31] == 12) or not np.all(grid[41, 31:64] == 12):
        return False
    
    # Check rows 44-46
    if not np.all(grid[44, 0:14] == 12) or not np.all(grid[44, 14:22] == 8) or not np.all(grid[44, 22:31] == 11) or not np.all(grid[44, 31:40] == 12) or not np.all(grid[44, 40:43] == 11) or not np.all(grid[44, 43:64] == 12):
        return False
    
    # Check rows 47-49
    if not np.all(grid[47, 0:17] == 12) or not np.all(grid[47, 17:28] == 11) or not np.all(grid[47, 28:31] == 12) or not np.all(grid[47, 31:64] == 12):
        return False
    
    # Check rows 50-52
    if not np.all(grid[50, 0:23] == 12) or not np.all(grid[50, 23:34] == 11) or not np.all(grid[50, 34:52] == 12) or not np.all(grid[50, 52:60] == 8) or not np.all(grid[50, 60:63] == 12) or not np.all(grid[50, 63:64] == 8) or not np.all(grid[50, 64:65] == 12):
        return False
    
    # Check rows 53-55
    if not np.all(grid[53, 0:26] == 12) or not np.all(grid[53, 26:34] == 8) or not np.all(grid[53, 34:55] == 12) or not np.all(grid[53, 55:64] == 9) or not np.all(grid[53, 64:73] == 12) or not np.all(grid[53, 73:76] == 9) or not np.all(grid[53, 76:81] == 12):
        return False
    
    # Check rows 56-58
    if not np.all(grid[56, 0:50] == 12) or not np.all(grid[56, 50:59] == 9) or not np.all(grid[56, 59:64] == 12):
        return False
    
    return True