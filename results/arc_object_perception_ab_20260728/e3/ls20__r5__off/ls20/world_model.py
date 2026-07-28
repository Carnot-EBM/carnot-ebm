import numpy as np

def parse_runs(row_str):
    """Parse a row run-length string into a list of (value, count) tuples."""
    runs = []
    parts = row_str.split(',')
    for part in parts:
        if not part:
            continue
        if 'x' in part:
            val_str, count_str = part.split('x')
            runs.append((int(val_str), int(count_str)))
        else:
            runs.append((int(part), 1))
    return runs

def grid_from_runs(rows_str_list):
    """Convert list of row run-length strings to a numpy grid."""
    grid = np.zeros((64, 64), dtype=int)
    for i, row_str in enumerate(rows_str_list):
        runs = parse_runs(row_str)
        col = 0
        for val, count in runs:
            grid[i, col:col+count] = val
            col += count
    return grid

def grid_to_runs(grid):
    """Convert a numpy grid to a list of row run-length strings."""
    rows = []
    for i in range(64):
        row = grid[i, :]
        if len(row) == 0:
            rows.append("")
            continue
        runs = []
        current_val = row[0]
        count = 1
        for j in range(1, len(row)):
            if row[j] == current_val:
                count += 1
            else:
                runs.append((current_val, count))
                current_val = row[j]
                count = 1
        runs.append((current_val, count))
        row_str = ','.join(f"{v}x{n}" for v, n in runs)
        rows.append(row_str)
    return rows

def apply_delta(grid, delta_str):
    """Apply a delta run-length string to the grid in place."""
    # Parse delta runs
    runs = []
    parts = delta_str.split(',')
    for part in parts:
        if not part:
            continue
        if 'x' in part:
            val_str, count_str = part.split('x')
            runs.append((int(val_str), int(count_str)))
        else:
            runs.append((int(part), 1))
    
    # Apply runs to grid
    for run in runs:
        val, count = run
        # Find the first empty cell in the row
        row_idx = -1
        for r in range(64):
            if grid[r, :] != grid[r, :].copy():
                row_idx = r
                break
        
        # This is a simplified approach - we need to handle the specific delta format
        # The delta format is: r<row>c<col0>:<v0>x<n0>,<v1>x<n1>,...
        # We need to parse this properly
        pass

def engine(grid, action, data):
    """
    Predict the next grid state based on the action.
    
    Actions:
    1: Move up
    2: Move down
    3: Move left
    4: Move right
    5: Move up-left
    6: Move up-right
    7: Move down-left
    
    The game appears to be a gravity-based puzzle where blocks fall and merge.
    """
    grid = grid.copy()
    
    # Determine the direction of movement based on action
    if action == 1:  # Up
        for col in range(64):
            for row in range(63, -1, -1):
                if grid[row, col] != 0:
                    # Try to move up
                    for new_row in range(row - 1, -1, -1):
                        if grid[new_row, col] == 0:
                            grid[new_row, col] = grid[row, col]
                            grid[row, col] = 0
                            break
    elif action == 2:  # Down
        for col in range(64):
            for row in range(63, -1, -1):
                if grid[row, col] != 0:
                    # Try to move down
                    for new_row in range(row + 1, 64):
                        if grid[new_row, col] == 0:
                            grid[new_row, col] = grid[row, col]
                            grid[row, col] = 0
                            break
    elif action == 3:  # Left
        for row in range(64):
            for col in range(63, -1, -1):
                if grid[row, col] != 0:
                    # Try to move left
                    for new_col in range(col - 1, -1, -1):
                        if grid[row, new_col] == 0:
                            grid[row, new_col] = grid[row, col]
                            grid[row, col] = 0
                            break
    elif action == 4:  # Right
        for row in range(64):
            for col in range(63, -1, -1):
                if grid[row, col] != 0:
                    # Try to move right
                    for new_col in range(col + 1, 64):
                        if grid[row, new_col] == 0:
                            grid[row, new_col] = grid[row, col]
                            grid[row, col] = 0
                            break
    elif action == 5:  # Up-left
        for row in range(64):
            for col in range(64):
                if grid[row, col] != 0:
                    # Try to move up-left
                    for new_row in range(row - 1, -1, -1):
                        for new_col in range(col - 1, -1, -1):
                            if grid[new_row, new_col] == 0:
                                grid[new_row, new_col] = grid[row, col]
                                grid[row, col] = 0
                                break
                        if grid[row, col] == 0:
                            break
    elif action == 6:  # Up-right
        for row in range(64):
            for col in range(64):
                if grid[row, col] != 0:
                    # Try to move up-right
                    for new_row in range(row - 1, -1, -1):
                        for new_col in range(col + 1, 64):
                            if grid[new_row, new_col] == 0:
                                grid[new_row, new_col] = grid[row, col]
                                grid[row, col] = 0
                                break
                        if grid[row, col] == 0:
                            break
    elif action == 7:  # Down-left
        for row in range(64):
            for col in range(64):
                if grid[row, col] != 0:
                    # Try to move down-left
                    for new_row in range(row + 1, 64):
                        for new_col in range(col - 1, -1, -1):
                            if grid[new_row, new_col] == 0:
                                grid[new_row, new_col] = grid[row, col]
                                grid[row, col] = 0
                                break
                        if grid[row, col] == 0:
                            break
    
    return grid

def is_level_complete(grid):
    """
    Check if the grid is in a win state.
    
    The win state appears to have a specific pattern where blocks are arranged
    in a particular way.
    """
    # Check if the grid matches the win state pattern
    # This is a simplified check - we need to identify the specific win condition
    return True