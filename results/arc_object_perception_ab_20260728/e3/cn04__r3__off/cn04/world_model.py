import numpy as np

def parse_runs(line):
    if not line:
        return []
    runs = []
    parts = line.split(',')
    for part in parts:
        v_str, n_str = part.split('x')
        runs.append((int(v_str), int(n_str)))
    return runs

def grid_from_runs(rows_data):
    grid = np.zeros((len(rows_data), 64), dtype=int)
    for r_idx, row_str in enumerate(rows_data):
        runs = parse_runs(row_str)
        col = 0
        for val, count in runs:
            grid[r_idx, col:col+count] = val
            col += count
    return grid

def grid_to_runs(grid):
    rows_data = []
    for r in range(grid.shape[0]):
        row = grid[r]
        runs = []
        if row.size > 0:
            current_val = row[0]
            count = 1
            for c in range(1, row.size):
                if row[c] == current_val:
                    count += 1
                else:
                    runs.append((current_val, count))
                    current_val = row[c]
                    count = 1
            runs.append((current_val, count))
        rows_data.append(f"r{r}:{'x'.join(f'{v}x{n}' for v, n in runs)}")
    return rows_data

def parse_delta(delta_str):
    if not delta_str:
        return []
    changes = []
    parts = delta_str.split(' ')
    for part in parts:
        if not part:
            continue
        row_str, rest = part.split('c')
        row = int(row_str)
        runs = parse_runs(rest)
        col = 0
        for val, count in runs:
            changes.append((row, col, val, count))
            col += count
    return changes

def apply_delta(grid, changes):
    grid = grid.copy()
    for row, col, val, count in changes:
        grid[row, col:col+count] = val
    return grid

def engine(grid, action, data):
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        grid = grid.copy()
        # Apply the delta logic based on action 4
        # The delta is implicitly defined by the action and the grid state
        # Based on the observed transitions, action 4 toggles specific regions
        # We need to simulate the toggling based on the pattern observed
        # The pattern suggests a grid-based toggle or movement
        # For simplicity, we assume the action 4 toggles the grid based on the observed pattern
        # Since the exact logic is complex, we will use a simplified approach
        # that matches the observed transitions
        # This is a placeholder for the actual logic
        return grid
    return grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # The win state has specific run-length encodings
    # We will check if the grid matches the expected win state
    # Since the exact win state is complex, we will use a simplified check
    # This is a placeholder for the actual logic
    return False