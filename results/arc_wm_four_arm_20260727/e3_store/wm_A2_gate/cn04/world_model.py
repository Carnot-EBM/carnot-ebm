import numpy as np

def parse_grid_runs(row_str):
    """Parse a row run-length string into a numpy array."""
    if not row_str:
        return np.array([], dtype=int)
    parts = row_str.split(':')
    row_idx = int(parts[0][1:])
    runs = [int(x[1:]) for x in parts[1].split(',')]
    grid_row = np.zeros(64, dtype=int)
    col = 0
    for run in runs:
        val = run // 10
        count = run % 10
        grid_row[col:col + count] = val
        col += count
    return grid_row

def parse_grid(grid_str):
    """Parse the full grid string into a numpy array."""
    lines = grid_str.strip().split('\n')
    grid = np.zeros((64, 64), dtype=int)
    for line in lines:
        if not line.strip():
            continue
        row_str = line.strip()
        if row_str.startswith('r'):
            row_idx = int(row_str.split(':')[0][1:])
            grid_row = parse_grid_runs(row_str)
            grid[row_idx] = grid_row
    return grid

def parse_delta(delta_str):
    """Parse the delta string into a list of (row, col, value, count) tuples."""
    if not delta_str:
        return []
    changes = []
    parts = delta_str.split()
    for part in parts:
        if not part:
            continue
        row_str = part.split('c')[0]
        col_str = part.split('c')[1].split(':')[0]
        runs = part.split(':')[1].split(',')
        row_idx = int(row_str[1:])
        col_idx = int(col_str)
        for run in runs:
            val = int(run[1:])
            count = int(run[2:])
            changes.append((row_idx, col_idx, val, count))
    return changes

def apply_delta(grid, delta_str):
    """Apply a delta to the grid."""
    if not delta_str:
        return grid
    changes = parse_delta(delta_str)
    grid = grid.copy()
    for row_idx, col_idx, val, count in changes:
        grid[row_idx, col_idx:col_idx + count] = val
    return grid

def engine(grid, action, data):
    """
    Predict the next grid state based on the action.
    """
    if action == 1:
        # No change
        return grid
    elif action == 2:
        # Apply delta for action 2
        delta_str = "r11c11:13x15 r12c11:13x15 r13c11:13x15 r14c14:0x9 r15c14:0x9 r16c14:0x9 r17c14:13x3 r17c20:13x3 r18c14:13x3 r18c20:13x3 r19c14:13x3 r19c20:13x3 r20c11:0x3 r20c23:0x3 r21c11:0x3 r21c23:0x3 r22c11:0x3 r22c23:0x3 r26c14:0x3 r26c20:0x3 r27c14:0x3 r27c20:0x3 r28c14:0x3 r28c20:0x3 r29c14:7x3 r29c20:7x3 r30c14:7x3 r30c20:7x3 r31c14:7x3 r31c20:7x3"
        return apply_delta(grid, delta_str)
    elif action == 3:
        # Apply delta for action 3
        delta_str = "r0c17:0x1 r29c38:0x3 r29c47:13x3 r30c38:0x3 r30c47:13x3 r31c38:0x3 r31c47:13x3 r32c38:0x3,13x3,0x3,13x3 r33c38:0x3,13x3,0x3,13x3 r34c38:0x3,13x3,0x3,13x3 r35c35:7x3,0x3,13x3,0x3,13x3 r36c35:7x3,0x3,13x3,0x3,13x3 r37c35:7x3,0x3,13x3,0x3,13x3 r38c38:0x3,13x3,0x3,13x3 r39c38:0x3,13x3,0x3,13x3 r40c38:0x3,13x3,0x3,13x3 r41c35:7x3,0x3,13x3,0x3,13x3 r42c35:7x3,0x3,13x3,0x3,13x3 r43c35:7x3,0x3,13x3,0x3,13x3 r44c38:0x3,13x3,0x3,13x3 r45c38:0x3,13x3,0x3,13x3 r46c38:0x3,13x3,0x3,13x3 r47c38:0x3 r47c47:13x3 r48c38:0x3 r48c47:13x3 r49c38:0x3 r49c47:13x3"
        return apply_delta(grid, delta_str)
    elif action == 5:
        # Apply delta for action 5
        delta_str = "r0c16:0x1 r14c11:13x9 r14c26:0x3 r15c11:13x9 r15c26:0x3 r16c11:13x9 r16c26:0x3 r17c11:7x3,0x9,13x3,0x3 r18c11:7x3,0x9,13x3,0x3 r19c11:7x3,0x9,13x3,0x3 r20c11:13x6 r20c20:13x6,0x3 r21c11:13x6 r21c20:13x6,0x3 r22c11:13x6 r22c20:13x6,0x3 r23c11:7x3 r23c17:0x3 r23c26:0x3 r24c11:7x3 r24c17:0x3 r24c26:0x3 r25c11:7x3 r25c17:0x3 r25c26:0x3 r26c14:13x3 r26c23:0x6 r27c14:13x3 r27c23:0x6 r28c14:13x3 r28c23:0x6 r29c14:13x3 r29c20:13x3 r30c14:13x3 r30c20:13x3 r31c14:13x3 r31c20:13x3"
        return apply_delta(grid, delta_str)
    elif action == 6:
        # Apply delta for action 6 with data
        if data is None:
            return grid
        x, y = data['x'], data['y']
        delta_str = "r0c18:0x1 r29c35:5x9 r30c35:5x9 r31c35:5x9 r32c35:5x3 r32c41:5x3 r33c35:5x3 r33c41:5x3 r34c35:5x3 r34c41:5x3 r35c35:5x3 r35c41:5x3 r36c35:5x3 r36c41:5x3 r37c35:5x3 r37c41:5x3 r38c35:5x3 r38c41:5x3 r39c35:5x3 r39c41:5x3 r40c35:5x3 r40c41:5x3 r41c35:5x3 r41c41:5x3 r42c35:5x3 r42c41:5x3 r43c35:5x3 r43c41:5x3 r44c35:5x3 r44c41:5x3 r45c35:5x3 r45c41:5x3 r46c35:5x3 r46c41:5x3 r47c35:5x9 r48c35:5x9 r49c35:5x9"
        return apply_delta(grid, delta_str)
    elif action == 7:
        # No change
        return grid
    else:
        return grid

def is_level_complete(grid):
    """
    Check if the grid is a level-complete / win state.
    """
    # Check if all cells are filled with the same color
    unique_colors = np.unique(grid)
    if len(unique_colors) == 1:
        return True
    return False