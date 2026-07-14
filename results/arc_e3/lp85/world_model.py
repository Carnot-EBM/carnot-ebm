import numpy as np

def parse_run_length_row(row_str):
    runs = []
    if not row_str:
        return runs
    parts = row_str.split(':')
    if len(parts) != 2:
        return runs
    row_idx = int(parts[0][1:])
    runs_str = parts[1]
    if not runs_str:
        return runs
    run_parts = runs_str.split(',')
    for rp in run_parts:
        val, count = rp.split('x')
        runs.append((int(val), int(count)))
    return runs

def parse_grid(grid_str):
    grid = np.zeros((64, 64), dtype=int)
    rows = grid_str.strip().split('\n')
    for row_line in rows:
        if not row_line:
            continue
        runs = parse_run_length_row(row_line)
        col = 0
        for val, count in runs:
            grid[row_idx, col:col+count] = val
            col += count
    return grid

def parse_delta(delta_str):
    if not delta_str:
        return []
    lines = delta_str.strip().split('\n')
    transitions = []
    for line in lines:
        if not line:
            continue
        parts = line.split(' ')
        if len(parts) < 2:
            continue
        row_idx = int(parts[0][1:])
        runs_str = parts[1]
        if not runs_str:
            continue
        run_parts = runs_str.split(',')
        for rp in run_parts:
            val, count = rp.split('x')
            transitions.append((row_idx, int(val), int(count)))
    return transitions

def apply_delta(grid, delta_runs):
    grid = grid.copy()
    for row_idx, val, count in delta_runs:
        grid[row_idx, :] = np.where(grid[row_idx, :] == 0, val, grid[row_idx, :])
    return grid

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        grid = grid.copy()
        grid[py, px] = 5
        return grid
    return grid

def is_level_complete(grid):
    grid = grid.copy()
    for i in range(64):
        if grid[i, 0] != 14:
            return False
    for i in range(64):
        if grid[i, 1] != 3:
            return False
    for i in range(11, 64):
        if grid[i, 1] != 3:
            return False
    for i in range(11, 64):
        if grid[i, 11] != 4:
            return False
    for i in range(11, 64):
        if grid[i, 12] != 3:
            return False
    return True