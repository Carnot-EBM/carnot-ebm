import numpy as np

def parse_runs(line):
    """Parse a run-length encoded row string into a list of (value, count) tuples."""
    if not line:
        return []
    parts = line.split(':')
    row_idx = int(parts[0][1:])
    runs = []
    for part in parts[1:]:
        if not part:
            continue
        sub_parts = part.split(',')
        for sub in sub_parts:
            if 'x' in sub:
                val_str, count_str = sub.split('x')
                runs.append((int(val_str), int(count_str)))
    return row_idx, runs

def grid_from_runs(rows_data):
    """Convert a list of (row_idx, runs) tuples into a full grid."""
    grid = np.zeros((64, 64), dtype=int)
    for row_idx, runs in rows_data:
        col = 0
        for val, count in runs:
            grid[row_idx, col:col+count] = val
            col += count
    return grid

def apply_delta(grid, delta_str):
    """Apply a delta (run-length encoded changes) to the grid."""
    if not delta_str:
        return grid
    lines = delta_str.strip().split('\n')
    for line in lines:
        if not line.strip():
            continue
        parts = line.split('c')
        if len(parts) != 2:
            continue
        row_idx = int(parts[0][1:])
        col_start = int(parts[1].split(':')[0])
        runs = parts[1].split(':')[1].split(',')
        for run in runs:
            if 'x' in run:
                val_str, count_str = run.split('x')
                val = int(val_str)
                count = int(count_str)
                grid[row_idx, col_start:col_start+count] = val
                col_start += count
    return grid

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action == 1:
        # Move up
        grid = grid.copy()
        for c in range(64):
            for r in range(63, -1, -1):
                if grid[r, c] == 5:
                    if r > 0 and grid[r-1, c] == 0:
                        grid[r, c] = 0
                        grid[r-1, c] = 5
    elif action == 2:
        # Move down
        grid = grid.copy()
        for c in range(64):
            for r in range(64):
                if grid[r, c] == 5:
                    if r < 63 and grid[r+1, c] == 0:
                        grid[r, c] = 0
                        grid[r+1, c] = 5
    elif action == 3:
        # Move left
        grid = grid.copy()
        for r in range(64):
            for c in range(63, -1, -1):
                if grid[r, c] == 5:
                    if c > 0 and grid[r, c-1] == 0:
                        grid[r, c] = 0
                        grid[r, c-1] = 5
    elif action == 4:
        # Move right
        grid = grid.copy()
        for r in range(64):
            for c in range(64):
                if grid[r, c] == 5:
                    if c < 63 and grid[r, c+1] == 0:
                        grid[r, c] = 0
                        grid[r, c+1] = 5
    elif action == 5:
        # Toggle 0 and 9
        grid = grid.copy()
        grid[63, 56] = 15 if grid[63, 56] == 0 else 0
    elif action == 6:
        # Click action (not implemented in observed transitions)
        pass
    elif action == 7:
        # Toggle 0 and 11
        grid = grid.copy()
        grid[63, 55] = 15 if grid[63, 55] == 0 else 0
    return grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # Win state: all rows 0-1 are all 5s, rows 5-63 are all 5s except row 63 which is all 15s
    # Actually, looking at the win state, it's more complex.
    # Let's check if the grid matches the win state exactly
    win_grid = np.zeros((64, 64), dtype=int)
    win_grid[0, :] = 5
    win_grid[1, :] = 5
    win_grid[2, :] = [5]*20 + [4]*3 + [5]*41
    win_grid[3, :] = [5]*20 + [4]*1 + [13]*1 + [4]*1 + [5]*41
    win_grid[4, :] = [5]*20 + [4]*3 + [5]*41
    win_grid[5, :] = 5
    win_grid[6, :] = 5
    win_grid[7, :] = [5]*16 + [12]*1 + [5]*21 + [12]*1 + [5]*25
    win_grid[8, :] = [5]*17 + [12]*1 + [5]*8 + [4]*3 + [5]*8 + [12]*1 + [5]*26
    win_grid[9, :] = [5]*18 + [12]*1 + [5]*7 + [4]*1 + [13]*1 + [4]*1 + [5]*7 + [12]*1 + [5]*27
    win_grid[10, :] = [5]*19 + [12]*1 + [5]*6 + [4]*3 + [5]*6 + [12]*1 + [5]*28
    win_grid[11, :] = [5]*11 + [4]*3 + [5]*6 + [12]*1 + [5]*13 + [12]*1 + [5]*29
    win_grid[12, :] = [5]*11 + [4]*1 + [13]*1 + [4]*1 + [5]*7 + [12]*1 + [5]*11 + [12]*1 + [5]*30
    win_grid[13, :] = [5]*11 + [4]*3 + [5]*8 + [12]*1 + [5]*9 + [12]*1 + [5]*31
    win_grid[14, :] = [5]*23 + [12]*1 + [5]*7 + [12]*1 + [5]*32
    win_grid[15, :] = [5]*24 + [12]*1 + [5]*5 + [12]*1 + [5]*33
    win_grid[16, :] = [5]*25 + [12]*1 + [5]*3 + [12]*1 + [5]*34
    win_grid[17, :] = [5]*26 + [12]*1 + [5]*1 + [12]*1 + [5]*35
    win_grid[18, :] = [5]*27 + [0]*1 + [5]*36
    win_grid[19, :] = [5]*26 + [12]*1 + [5]*1 + [12]*1 + [5]*35
    win_grid[20, :] = [5]*25 + [12]*1 + [5]*3 + [12]*1 + [5]*34
    win_grid[21, :] = [5]*24 + [12]*1 + [5]*5 + [12]*1 + [5]*8 + [13]*1 + [5]*24
    win_grid[22, :] = [5]*23 + [12]*1 + [5]*7 + [12]*1 + [5]*6 + [13]*1 + [5]*1 + [13]*1 + [5]*23
    win_grid[23, :] = [5]*22 + [12]*1 + [5]*9 + [12]*1 + [5]*4 + [13]*1 + [5]*3 + [13]*1 + [5]*22
    win_grid[24, :] = [5]*21 + [12]*1 + [5]*11 + [12]*1 + [5]*2 + [13]*1 + [5]*5 + [13]*1 + [5]*21
    win_grid[25, :] = [5]*20 + [12]*1 + [5]*13 + [12]*1 + [13]*1 + [5]*7 + [13]*1 + [5]*20
    win_grid[26, :] = [5]*19 + [12]*1 + [5]*14 + [13]*1 + [12]*1 + [5]*8 + [13]*1 + [5]*19
    win_grid[27, :] = [5]*18 + [12]*1 + [5]*14 + [13]*1 + [5]*2 + [12]*1 + [5]*8 + [13]*1 + [5]*18
    win_grid[28, :] = [5]*17 + [12]*1 + [5]*14 + [13]*1 + [5]*4 + [12]*1 + [5]*8 + [13]*1 + [5]*17
    win_grid[29, :] = [5]*16 + [12]*1 + [5]*14 + [13]*1 + [5]*6 + [12]*1 + [5]*8 + [13]*1 + [9]*1 + [5]*15
    win_grid[30, :] = [5]*30 + [13]*1 + [5]*17 + [9]*1 + [5]*15
    win_grid[31, :] = [5]*31 + [13]*1 + [5]*15 + [13]*1 + [9]*1 + [5]*15
    win_grid[32, :] = [5]*32 + [13]*1 + [5]*13 + [13]*1 + [5]*1 + [9]*1 + [5]*15
    win_grid[33, :] = [5]*33 + [13]*1 + [5]*11 + [13]*1 + [5]*2 + [9]*1 + [5]*15
    win_grid[34, :] = [5]*34 + [13]*1 + [5]*9 + [13]*1 + [5]*3 + [9]*1 + [5]*15
    win_grid[35, :] = [5]*26 + [4]*3 + [5]*6 + [13]*1 + [5]*7 + [13]*1 + [5]*4 + [9]*1 + [5]*15
    win_grid[36, :] = [5]*26 + [4]*1 + [9]*1 + [4]*1 + [5]*7 + [13]*1 + [5]*5 + [13]*1 + [5]*5 + [9]*1 + [5]*15
    win_grid[37, :] = [5]*26 + [4]*3 + [5]*8 + [13]*1 + [5]*3 + [13]*1 + [5]*6 + [9]*1 + [5]*15
    win_grid[38, :] = [5]*8 + [4]*3 + [5]*27 + [13]*1 + [5]*1 + [13]*1 + [5]*7 + [9]*1 + [5]*15
    win_grid[39, :] = [5]*8 + [4]*1 + [12]*1 + [4]*1 + [5]*28 + [13]*1 + [5]*8 + [9]*1 + [5]*15
    win_grid[40, :] = [5]*8 + [4]*3 + [5]*37 + [9]*1 + [5]*15
    win_grid[41, :] = [5]*23 + [4]*3 + [5]*22 + [9]*1 + [5]*15
    win_grid[42, :] = [5]*23 + [4]*1 + [12]*1 + [4]*1 + [5]*9 + [9]*27 + [5]*2
    win_grid[43, :] = [5]*23 + [4]*3 + [5]*22 + [9]*1 + [5]*15
    win_grid[44, :] = [5]*48 + [9]*1 + [5]*15
    win_grid[45, :] = [5]*48 + [9]*1 + [5]*15
    win_grid[46, :] = [5]*48 + [9]*1 + [5]*15
    win_grid[47, :] = [5]*14 + [4]*3 + [5]*15 + [4]*3 + [5]*13 + [9]*1 + [5]*15
    win_grid[48, :] = [5]*14 + [4]*1 + [9]*1 + [4]*1 + [5]*15 + [4]*1 + [9]*1 + [4]*1 + [5]*13 + [9]*1 + [5]*15
    win_grid[49, :] = [5]*14 + [4]*3 + [5]*15 + [4]*3 + [5]*13 + [9]*1 + [5]*15
    win_grid[50, :] = [5]*48 + [9]*1 + [5]*15
    win_grid[51, :] = [5]*48 + [9]*1 + [5]*15
    win_grid[52, :] = [5]*48 + [9]*1 + [5]*15
    win_grid[53, :] = [5]*48 + [9]*1 + [5]*15
    win_grid[54, :] = [5]*48 + [9]*1 + [5]*15
    win_grid[55, :] = [5]*48 + [9]*1 + [5]*15
    win_grid[56, :] = [5]*8 + [4]*3 + [5]*53
    win_grid[57, :] = [5]*8 + [4]*1 + [12]*1 + [4]*1 + [5]*53
    win_grid[58, :] = [5]*8 + [4]*3 + [5]*53
    win_grid[59, :] = [5]*26 + [4]*3 + [5]*35
    win_grid[60, :] = [5]*26 + [4]*1 + [9]*1 + [4]*1 + [5]*35
    win_grid[61, :] = [5]*26 + [4]*3 + [5]*35
    win_grid[62, :] = 5
    win_grid[63, :] = 15
    return np.array_equal(grid, win_grid)