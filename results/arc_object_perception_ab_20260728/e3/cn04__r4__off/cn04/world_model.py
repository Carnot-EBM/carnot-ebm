import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 4:
        if data is None:
            return new_grid
            
        # Parse the run-length encoded delta
        # Format: "r<row>c<col0>:<v0>x<n0>,<v1>x<n1>,..."
        # We need to apply these changes to the grid
        
        # Split by rows
        rows_data = data.split('\n')
        for row_str in rows_data:
            if not row_str.strip():
                continue
                
            # Parse row data
            # Format: "r<row>c<col0>:<v0>x<n0>,<v1>x<n1>,..."
            parts = row_str.strip().split('c')
            if len(parts) < 2:
                continue
                
            row_idx = int(parts[0].replace('r', ''))
            col_start = int(parts[1].split(':')[0])
            rest = parts[1].split(':')[1]
            
            # Parse runs
            runs = rest.split(',')
            col = col_start
            
            for run in runs:
                if not run.strip():
                    continue
                # Format: "<v>x<n>"
                val_count = run.split('x')
                val = int(val_count[0])
                count = int(val_count[1])
                
                # Apply the change
                new_grid[row_idx, col:col+count] = val
                col += count
                
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    
    # Check if the grid matches the win state pattern
    # The win state has specific run-length patterns
    
    # Check row 0
    row0 = grid[0]
    if not np.array_equal(row0, np.array([12]*16 + [4]*32 + [12]*16)):
        return False
    
    # Check rows 1-10
    for i in range(1, 11):
        if not np.array_equal(grid[i], np.ones(W) * 12):
            return False
    
    # Check rows 11-13
    for i in range(11, 14):
        if not np.array_equal(grid[i], np.array([12]*11 + [0]*9 + [12]*44)):
            return False
    
    # Check rows 14-16
    for i in range(14, 17):
        if not np.array_equal(grid[i], np.array([12]*11 + [0]*3 + [12]*3 + [0]*3 + [12]*24 + [8]*3 + [14]*3 + [12]*14)):
            return False
    
    # Check rows 17-19
    for i in range(17, 20):
        if not np.array_equal(grid[i], np.array([12]*11 + [0]*3 + [12]*3 + [0]*3 + [12]*27 + [14]*3 + [12]*14)):
            return False
    
    # Check rows 20-22
    for i in range(20, 23):
        if not np.array_equal(grid[i], np.array([12]*11 + [0]*3 + [12]*3 + [8]*3 + [12]*27 + [14]*3 + [8]*3 + [12]*11)):
            return False
    
    # Check rows 23-25
    for i in range(23, 26):
        if not np.array_equal(grid[i], np.array([12]*11 + [0]*3 + [12]*33 + [14]*3 + [12]*14)):
            return False
    
    # Check rows 26-28
    for i in range(26, 29):
        if not np.array_equal(grid[i], np.array([12]*11 + [0]*3 + [8]*3 + [12]*30 + [14]*3 + [8]*3 + [12]*11)):
            return False
    
    # Check rows 29-31
    for i in range(29, 32):
        if not np.array_equal(grid[i], np.array([12]*38 + [8]*3 + [12]*6 + [14]*3 + [12]*14)):
            return False
    
    # Check rows 32-34
    for i in range(32, 35):
        if not np.array_equal(grid[i], np.array([12]*38 + [14]*12 + [12]*14)):
            return False
    
    # Check rows 35-37
    for i in range(35, 38):
        if not np.array_equal(grid[i], np.array([12]*23 + [11]*9 + [12]*32)):
            return False
    
    # Check rows 38-40
    for i in range(38, 41):
        if not np.array_equal(grid[i], np.array([12]*17 + [8]*3 + [12]*3 + [11]*3 + [12]*3 + [11]*3 + [8]*3 + [12]*29)):
            return False
    
    # Check rows 41-43
    for i in range(41, 44):
        if not np.array_equal(grid[i], np.array([12]*17 + [11]*9 + [12]*3 + [11]*3 + [12]*32)):
            return False
    
    # Check rows 44-46
    for i in range(44, 47):
        if not np.array_equal(grid[i], np.array([12]*14 + [8]*3 + [11]*3 + [12]*9 + [11]*3 + [12]*32)):
            return False
    
    # Check rows 47-49
    for i in range(47, 50):
        if not np.array_equal(grid[i], np.array([12]*17 + [11]*9 + [12]*3 + [11]*3 + [12]*32)):
            return False
    
    # Check rows 50-52
    for i in range(50, 53):
        if not np.array_equal(grid[i], np.array([12]*23 + [11]*9 + [12]*18 + [8]*3 + [12]*3 + [8]*3 + [12]*5)):
            return False
    
    # Check rows 53-55
    for i in range(53, 56):
        if not np.array_equal(grid[i], np.array([12]*26 + [8]*3 + [12]*21 + [9]*3 + [12]*3 + [9]*3 + [12]*5)):
            return False
    
    # Check rows 56-58
    for i in range(56, 59):
        if not np.array_equal(grid[i], np.array([12]*50 + [9]*9 + [12]*5)):
            return False
    
    # Check rows 59-63
    for i in range(59, 64):
        if not np.array_equal(grid[i], np.ones(W) * 12):
            return False
    
    return True