import numpy as np

def parse_grid_line(line):
    """Parse a run-length encoded row string into a list of values."""
    if not line:
        return []
    parts = line.split(':')
    row_idx = int(parts[0][1:])
    runs = parts[1].split(',')
    result = []
    for run in runs:
        val, count = run.split('x')
        result.extend([int(val)] * int(count))
    return result

def parse_grid(grid_str):
    """Parse the full grid string into a numpy array."""
    lines = grid_str.strip().split('\n')
    grid = np.zeros((64, 64), dtype=int)
    for line in lines:
        if not line.strip():
            continue
        values = parse_grid_line(line)
        grid[:] = values
    return grid

def apply_delta(grid, delta_str):
    """Apply a delta (run-length encoded changes) to the grid."""
    if not delta_str.strip():
        return grid
    lines = delta_str.strip().split('\n')
    for line in lines:
        if not line.strip():
            continue
        parts = line.split('c')
        row_idx = int(parts[0][1:])
        col_start = int(parts[1][1:])
        runs = parts[2].split(',')
        for run in runs:
            val, count = run.split('x')
            val = int(val)
            count = int(count)
            grid[row_idx, col_start:col_start + count] = val
    return grid

def engine(grid, action, data):
    """
    Predict the next grid state based on the current grid and action.
    This is a placeholder implementation.
    """
    if action == 4:
        # Action 4 is a click action with data={'x':px, 'y':py}
        if data:
            x, y = data['x'], data['y']
            # Simulate a click effect (e.g., toggle a cell)
            grid[y, x] = 1 - grid[y, x]
    return grid

def is_level_complete(grid):
    """
    Check if the grid is in a win state.
    This is a placeholder implementation.
    """
    # Check if the grid matches the win state pattern
    win_grid = np.array([
        [12]*16 + [4]*32 + [12]*16,
        [12]*64,
        [12]*64,
        [12]*64,
        [12]*64,
        [12]*64,
        [12]*64,
        [12]*64,
        [12]*64,
        [12]*64,
        [12]*64,
        [12]*11 + [0]*9 + [12]*44,
        [12]*11 + [0]*9 + [12]*44,
        [12]*11 + [0]*9 + [12]*44,
        [12]*11 + [0]*3 + [12]*3 + [0]*3 + [12]*24 + [8]*3 + [14]*3 + [12]*14,
        [12]*11 + [0]*3 + [12]*3 + [0]*3 + [12]*24 + [8]*3 + [14]*3 + [12]*14,
        [12]*11 + [0]*3 + [12]*3 + [0]*3 + [12]*24 + [8]*3 + [14]*3 + [12]*14,
        [12]*11 + [0]*3 + [12]*3 + [0]*3 + [12]*27 + [14]*3 + [12]*14,
        [12]*11 + [0]*3 + [12]*3 + [0]*3 + [12]*27 + [14]*3 + [12]*14,
        [12]*11 + [0]*3 + [12]*3 + [0]*3 + [12]*27 + [14]*3 + [12]*14,
        [12]*11 + [0]*3 + [12]*3 + [8]*3 + [12]*27 + [14]*3 + [8]*3 + [12]*11,
        [12]*11 + [0]*3 + [12]*3 + [8]*3 + [12]*27 + [14]*3 + [8]*3 + [12]*11,
        [12]*11 + [0]*3 + [12]*3 + [8]*3 + [12]*27 + [14]*3 + [8]*3 + [12]*11,
        [12]*11 + [0]*3 + [12]*33 + [14]*3 + [12]*14,
        [12]*11 + [0]*3 + [12]*33 + [14]*3 + [12]*14,
        [12]*11 + [0]*3 + [12]*33 + [14]*3 + [12]*14,
        [12]*11 + [0]*3 + [8]*3 + [12]*30 + [14]*3 + [8]*3 + [12]*11,
        [12]*11 + [0]*3 + [8]*3 + [12]*30 + [14]*3 + [8]*3 + [12]*11,
        [12]*11 + [0]*3 + [8]*3 + [12]*30 + [14]*3 + [8]*3 + [12]*11,
        [12]*38 + [8]*3 + [12]*6 + [14]*3 + [12]*14,
        [12]*38 + [8]*3 + [12]*6 + [14]*3 + [12]*14,
        [12]*38 + [8]*3 + [12]*6 + [14]*3 + [12]*14,
        [12]*38 + [14]*12 + [12]*14,
        [12]*38 + [14]*12 + [12]*14,
        [12]*38 + [14]*12 + [12]*14,
        [12]*23 + [11]*9 + [12]*32,
        [12]*23 + [11]*9 + [12]*32,
        [12]*23 + [11]*9 + [12]*32,
        [12]*17 + [8]*3 + [12]*3 + [11]*3 + [12]*3 + [11]*3 + [8]*3 + [12]*29,
        [12]*17 + [8]*3 + [12]*3 + [11]*3 + [12]*3 + [11]*3 + [8]*3 + [12]*29,
        [12]*17 + [8]*3 + [12]*3 + [11]*3 + [12]*3 + [11]*3 + [8]*3 + [12]*29,
        [12]*17 + [11]*9 + [12]*3 + [11]*3 + [12]*32,
        [12]*17 + [11]*9 + [12]*3 + [11]*3 + [12]*32,
        [12]*17 + [11]*9 + [12]*3 + [11]*3 + [12]*32,
        [12]*14 + [8]*3 + [11]*3 + [12]*9 + [11]*3 + [12]*32,
        [12]*14 + [8]*3 + [11]*3 + [12]*9 + [11]*3 + [12]*32,
        [12]*14 + [8]*3 + [11]*3 + [12]*9 + [11]*3 + [12]*32,
        [12]*17 + [11]*9 + [12]*3 + [11]*3 + [12]*32,
        [12]*17 + [11]*9 + [12]*3 + [11]*3 + [12]*32,
        [12]*17 + [11]*9 + [12]*3 + [11]*3 + [12]*32,
        [12]*23 + [11]*9 + [12]*18 + [8]*3 + [12]*3 + [8]*3 + [12]*5,
        [12]*23 + [11]*9 + [12]*18 + [8]*3 + [12]*3 + [8]*3 + [12]*5,
        [12]*23 + [11]*9 + [12]*18 + [8]*3 + [12]*3 + [8]*3 + [12]*5,
        [12]*26 + [8]*3 + [12]*21 + [9]*3 + [12]*3 + [9]*3 + [12]*5,
        [12]*26 + [8]*3 + [12]*21 + [9]*3 + [12]*3 + [9]*3 + [12]*5,
        [12]*26 + [8]*3 + [12]*21 + [9]*3 + [12]*3 + [9]*3 + [12]*5,
        [12]*50 + [9]*9 + [12]*5,
        [12]*50 + [9]*9 + [12]*5,
        [12]*50 + [9]*9 + [12]*5,
        [12]*64,
        [12]*64,
        [12]*64,
        [12]*64,
        [12]*64,
        [12]*64,
        [12]*64,
        [12]*64
    ])
    return np.array_equal(grid, win_grid)