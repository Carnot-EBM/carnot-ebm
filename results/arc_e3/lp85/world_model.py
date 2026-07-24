import numpy as np

def parse_runs(row_str):
    """Parse a run-length encoded row string into a list of (value, count) tuples."""
    runs = []
    parts = row_str.split(':')
    for part in parts:
        if not part:
            continue
        runs.append([int(x) for x in part.split(',')])
    return runs

def parse_grid(grid_str):
    """Parse the full grid string into a numpy array."""
    grid = np.zeros((64, 64), dtype=np.int32)
    for i, row_str in enumerate(grid_str.split('\n')):
        if not row_str.strip():
            continue
        runs = parse_runs(row_str)
        col = 0
        for value, count in runs:
            grid[i, col:col+count] = value
            col += count
    return grid

def parse_action_delta(action_str):
    """Parse the action delta string into a list of (row, col_start, value, count) tuples."""
    deltas = []
    parts = action_str.split(' r')
    for part in parts:
        if not part:
            continue
        row, rest = part.split('c', 1)
        row = int(row)
        runs = [int(x) for x in rest.split(',')]
        col = 0
        for value, count in runs:
            deltas.append((row, col, value, count))
            col += count
    return deltas

def apply_delta(grid, deltas):
    """Apply a set of deltas to the grid."""
    grid = grid.copy()
    for row, col, value, count in deltas:
        grid[row, col:col+count] = value
    return grid

def engine(grid, action, data):
    if action == 0:
        grid = apply_delta(grid, parse_action_delta(data))
    elif action == 1:
        grid = apply_delta(grid, parse_action_delta(data))
    elif action == 2:
        grid = apply_delta(grid, parse_action_delta(data))
    elif action == 3:
        grid = apply_delta(grid, parse_action_delta(data))
    elif action == 4:
        grid = apply_delta(grid, parse_action_delta(data))
    elif action == 5:
        grid = apply_delta(grid, parse_action_delta(data))
    elif action == 6:
        grid = apply_delta(grid, parse_action_delta(data))
    elif action == 7:
        grid = apply_delta(grid, parse_action_delta(data))
    return grid

def is_level_complete(grid):
    # Convert grid to string format
    grid_str = ""
    for i in range(64):
        row_str = ""
        runs = []
        current_val = grid[i, 0]
        count = 1
        for j in range(1, 64):
            if grid[i, j] == current_val:
                count += 1
            else:
                runs.append((current_val, count))
                current_val = grid[i, j]
                count = 1
        runs.append((current_val, count))
        row_str = f"r{i}:" + ",".join([f"{v}x{c}" for v, c in runs])
        grid_str += row_str + "\n"
    # Check against the win state
    win_state_str = """r0:14x1,3x63
r1:14x1,3x9,14x4,3x1,5x4,3x1,5x4,3x1,5x4,3x1,5x4,3x1,5x4,3x1,5x4,3x1,5x4,3x15
r2:14x1,3x63
r3:14x1,3x63
r4:14x1,3x63
r5:14x1,3x63
r6:14x1,3x63
r7:14x1,3x63
r8:14x1,3x63
r9:14x1,3x63
r10:14x1,3x63
r11:14x1,3x10,4x41,3x12
r12:14x1,3x10,4x41,3x12
r13:14x1,3x10,4x41,3x12
r14:14x1,3x10,4x41,3x12
r15:14x1,3x10,4x41,3x12
r16:14x1,3x10,4x9,8x2,4x16,14x2,4x12,3x12
r17:14x1,3x10,4x8,8x3,4x1,10x2,4x1,9x2,4x1,15x2,4x1,11x2,4x1,15x2,4x1,14x3,4x11,3x12
r18:14x1,3x10,4x8,8x3,4x1,10x2,4x1,9x2,4x1,15x2,4x1,11x2,4x1,15x2,4x1,14x3,4x11,3x12
r19:14x1,3x10,4x9,8x2,4x16,14x2,4x12,3x12
r20:14x1,3x10,4x12,15x2,4x10,9x2,4x15,3x12
r21:14x1,3x10,4x12,15x2,4x10,9x2,4x15,3x12
r22:14x1,3x10,4x41,3x12
r23:14x1,3x10,4x12,2x2,4x10,10x2,4x15,3x12
r24:14x1,3x10,4x12,2x2,4x10,10x2,4x15,3x12
r25:14x1,3x10,4x3,8x2,4x18,11x1,4x2,11x1,4x9,14x2,4x3,3x12
r26:14x1,3x10,4x2,8x3,4x1,1x2,4x1,9x2,4x1,9x2,4x1,1x2,4x1,10x2,4x1,15x2,4x1,2x2,4x1,10x2,4x1,2x2,4x1,2x2,4x1,14x3,4x2,3x12
r27:14x1,3x10,4x2,8x3,4x1,1x2,4x1,9x2,4x1,9x2,4x1,1x2,4x1,10x2,4x1,15x2,4x1,2x2,4x1,10x2,4x1,2x2,4x1,2x2,4x1,14x3,4x2,3x12
r28:14x1,3x10,4x3,8x2,4x18,11x1,4x2,11x1,4x9,14x2,4x3,3x12
r29:14x1,3x10,4x12,1x2,4x10,1x2,4x15,3x12
r30:14x1,3x10,4x12,1x2,4x10,1x2,4x15,3x12
r31:14x1,3x1 state
r32:14x1,3x10,4x12,1x2,4x10,2x2,4x15,3x12
r33:14x1,3x10,4x12,1x2,4x10,2x2,4x15,3x12
r34:14x1,3x10,4x3,8x2,4x18,11x1,4x2,11x1,4x9,14x2,4x3,3x12
r35:14x1,3x10,4x2,8x3,4x1,1x2,4x1,10x2,4x1,9x2,4x1,1x2,4x1,9x2,4x1,10x2,4x1,15x2,4x1,9x2,4x1,2x2,4x1,10x2,4x1,14x3,4x2,3x12
r36:14x1,3x10,4x2,8x3,4x1,1x2,4x1,10x2,4x1,9x2,4x1,1x2,4x1,9x2,4x1,10x2,4x1,15x2,4x1,9x2,4x1,2x2,4x1,10x2,4x1,14x3,4x2,3x12
r37:14x1,3x10,4x3,8x2,4x18,11x1,4x2,11x1,4x9,14x2,4x3,3x12
r38:14x1,3x10,4x12,11x2,4x10,1x2,4x15,3x12
r39:14x1,3x10,4x12,11x2,4x10,1x2,4x15,3x12
r40:14x1,3x10,4x41,3x12
r41:14x1,3x10,4x12,2x2,4x10,9x2,4x15,3x12
r42:14x1,3x10,4x12,2x2,4x10,9x2,4x15,3x12
r43:14x1,3x10,4x41,3x12
r44:14x1,3x10,4x12,10x2,4x1,10x2,4x1,15x2,4x1,15x2,4x1,2x2,4x15,3x12
r45:14x1,3x10,4x12,10x2,4x1,10x2,4x1,15x2,4x1,15x2,4x1,2x2,4x15,3x12
r46:14x1,3x10,4x41,3x12
r47:14x1,3x10,4x41,3x12
r48:14x1,3x10,4x41,3x12
r49:14x1,3x10,4x41,3x12
r50:14x1,3x10,4x41,3x12
r51:14x1,3x10,4x41,3x12
r52:14x1,3x63
r53:14x1,3x63
r54:14x1,3x63
r55:14x1,3x63
r56:14x1,3x63
r57:14x1,3x63
r58:14x1,3x63
r59:14x1,3x63
r60:14x1,3x63
r61:14x1,3x63
r62:14x1,3x63
r63:14x1,3x63"""
    # Check if the grid string matches the win state string
    return grid_str == win_state_str