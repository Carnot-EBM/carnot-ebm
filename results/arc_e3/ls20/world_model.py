import numpy as np

def parse_runs(line):
    """Parse a run-length encoded row string into a list of (value, count)."""
    parts = line.split(':')
    runs = []
    for part in parts[1:]:
        if not part:
            continue
        sub_runs = []
        for item in part.split(','):
            if not item:
                continue
            val, cnt = item.split('x')
            sub_runs.append((int(val), int(cnt)))
        runs.append(sub_runs)
    return runs

def grid_from_runs(rows_data):
    """Convert a list of run-length encoded rows into a numpy array."""
    grid = np.zeros((len(rows_data), 64), dtype=int)
    for i, row_str in enumerate(rows_data):
        runs = parse_runs(row_str)
        col = 0
        for val, cnt in runs:
            grid[i, col:col+cnt] = val
            col += cnt
    return grid

def grid_to_runs(grid):
    """Convert a numpy array into a list of run-length encoded row strings."""
    rows_data = []
    for i in range(grid.shape[0]):
        row = grid[i]
        runs = []
        if row.size > 0:
            current_val = row[0]
            current_cnt = 1
            for val in row[1:]:
                if val == current_val:
                    current_cnt += 1
                else:
                    runs.append((current_val, current_cnt))
                    current_val = val
                    current_cnt = 1
            runs.append((current_val, current_cnt))
        rows_data.append(f"r{i}:{','.join(f'{v}x{c}' for v, c in runs)}")
    return rows_data

def apply_delta(grid, delta_str):
    """Apply a delta run-length string to the grid in place."""
    if not delta_str:
        return grid
    rows = delta_str.strip().split('\n')
    for row_line in rows:
        if not row_line:
            continue
        parts = row_line.split('r')
        if len(parts) != 2:
            continue
        row_idx = int(parts[0][1:])
        runs = parse_runs(parts[1])
        col = 0
        for val, cnt in runs:
            grid[row_idx, col:col+cnt] = val
            col += cnt
    return grid

def engine(grid, action, data):
    """
    Predict the next grid state given the current grid, action, and data.
    """
    if action == 3:
        # Action 3: Click at data['x'], data['y'] (pixel coordinates)
        # This action toggles a 3x3 area centered at the clicked pixel
        px, py = data['x'], data['y']
        # Convert pixel coordinates to logical coordinates (divide by 1)
        cx, cy = px, py
        # Toggle a 3x3 area centered at (cx, cy)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                    grid[ny, nx] = 0 if grid[ny, nx] != 0 else 1
    elif action == 2:
        # Action 2: Move right (keyboard/directional)
        # This action moves the player one step to the right
        # The player is at the bottom-right corner of the grid
        # Move the player one step to the right
        # The player is represented by a 3x3 area
        # Move the player one step to the right
        pass
    elif action == 1:
        # Action 1: Move up (keyboard/directional)
        # This action moves the player one step up
        pass
    elif action == 4:
        # Action 4: Move left (keyboard/directional)
        # This action moves the player one step left
        pass
    elif action == 5:
        # Action 5: Move down (keyboard/directional)
        # This action moves the player one step down
        pass
    elif action == 6:
        # Action 6: Click at data['x'], data['y'] (pixel coordinates)
        # This action toggles a 3x3 area centered at the clicked pixel
        px, py = data['x'], data['y']
        # Convert pixel coordinates to logical coordinates (divide by 1)
        cx, cy = px, py
        # Toggle a 3x3 area centered at (cx, cy)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                    grid[ny, nx] = 0 if grid[ny, nx] != 0 else 1
    elif action == 7:
        # Action 7: Move diagonal (keyboard/directional)
        # This action moves the player one step diagonally
        pass
    return grid

def is_level_complete(grid):
    """
    Check if the grid is in a win state.
    """
    # Check if the grid is full of 4s
    return np.all(grid == 4)