import numpy as np

def engine(grid, action, data):
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        # Apply gravity: objects fall down
        for x in range(w):
            stack = []
            for y in range(h - 1, -1, -1):
                if grid[y, x] != 0:
                    stack.append(grid[y, x])
            for i, v in enumerate(stack):
                new_grid[h - 1 - i, x] = v
            for y in range(h - 1 - len(stack), -1, -1):
                new_grid[y, x] = 0
        return new_grid
    elif action == 2:
        if data is None:
            return grid
        # Action 2: Move left
        h, w = grid.shape
        new_grid = grid.copy()
        for y in range(h):
            row = grid[y, :].copy()
            # Remove 0s
            non_zero = row[row != 0]
            # Pad with 0s on right
            new_row = np.zeros(w, dtype=int)
            new_row[:len(non_zero)] = non_zero
            new_grid[y, :] = new_row
        return new_grid
    elif action == 3:
        if data is None:
            return grid
        # Action 3: Move right
        h, w = grid.shape
        new_grid = grid.copy()
        for y in range(h):
            row = grid[y, :].copy()
            non_zero = row[row != 0]
            new_row = np.zeros(w, dtype=int)
            new_row[-len(non_zero):] = non_zero
            new_grid[y, :] = new_row
        return new_grid
    elif action == 5:
        if data is None:
            return grid
        # Action 5: Move up
        h, w = grid.shape
        new_grid = grid.copy()
        for x in range(w):
            col = grid[:, x].copy()
            non_zero = col[col != 0]
            new_col = np.zeros(h, dtype=int)
            new_col[:len(non_zero)] = non_zero
            new_grid[:, x] = new_col
        return new_grid
    elif action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        # Toggle cell
        if 0 <= py < h and 0 <= px < w:
            new_grid[py, px] = 15 - new_grid[py, px]
        return new_grid
    elif action == 7:
        if data is None:
            return grid
        # Action 7: Move down
        h, w = grid.shape
        new_grid = grid.copy()
        for x in range(w):
            col = grid[:, x].copy()
            non_zero = col[col != 0]
            new_col = np.zeros(h, dtype=int)
            new_col[-len(non_zero):] = non_zero
            new_grid[:, x] = new_col
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    h, w = grid.shape
    # Check if grid matches the win state pattern
    # The win state has specific patterns in rows 0-17 and 24-32
    # We check for the presence of specific structures
    # This is a simplified check based on the win state description
    # Check if the grid has the same structure as the win state
    # The win state has objects in specific positions
    # We check if the grid matches the win state exactly
    return np.array_equal(grid, win_state_grid)

# Define the win state grid
def get_win_state():
    win_rows = [
        "r0:5x16,4x2,3x46",
        "r1:5x16,4x2,3x46",
        "r2:5x16,4x2,3x14,4x5,3x1,4x5,3x1,4x5,3x15",
        "r3:5x3,15x9,12x1,5x3,4x2,3x14,4x1,0x3,4x1,3x1,4x1,15x3,4x1,3x1,4x1,12x3,4x1,3x15",
        "r4:5x3,15x8,12x2,5x3,4x2,3x14,4x1,0x3,4x1,3x1,4x1,15x3,4x1,3x1,4x1,12x3,4x1,3x15",
        "r5:5x3,15x7,12x3,5x3,4x2,3x14,4x1,0x3,4x1,3x1,4x1,15x3,4x1,3x1,4x1,12x3,4x1,3x15",
        "r6:5x3,15x6,12x4,5x3,4x2,3x14,4x5,3x1,4x5,3x1,4x5,3x15",
        "r7:5x3,15x5,12x5,5x3,4x2,3x20,0x5,3x21",
        "r8:5x3,0x4,12x6,5x3,4x2,3x46",
        "r9:5x3,0x3,12x7,5x3,4x2,5x46",
        "r10:5x3,0x2,12x8,5x3,4x2,5x46",
        "r11:5x3,0x1,12x9,5x3,4x2,5x46",
        "r12:5x3,12x10,5x3,4x2,5x46",
        "r13:5x16,4x2,5x46",
        "r14:5x16,4x2,5x46",
        "r15:5x16,4x2,5x46",
        "r16:4x18,5x46",
        "r17:4x18,5x46",
        "r18:5x64",
        "r19:5x64",
        "r20:5x64",
        "r21:5x64",
        "r22:5x64",
        "r23:5x64",
        "r24:5x25,2x14,5x25",
        "r25:5x25,2x1,15x12,2x1,5x25",
        "r26:5x25,2x1,15x12,2x1,5x25",
        "r27:5x25,2x1,15x12,2x1,5x25",
        "r28:5x25,2x1,15x12,2x1,5x25",
        "r29:5x25,2x1,15x12,2x1,5x25",
        "r30:5x25,2x1,15x12,2x1,5x25",
        "r31:5x25,2x1,15x12,2x1,5x25",
        "r32:5x25,2x1,5x12,2x1,5x25",
        "r33:5x64",
        "r34:5x27,0x10,5x27",
        "r35:5x27,0x10,5x27",
        "r36:5x27,0x10,5x27",
        "r37:5x27,0x10,5x27",
        "r38:5x27,0x10,5x27",
        "r39:5x27,0x10,5x27",
        "r40:5x27,0x10,5x27",
        "r41:5x27,0x10,5x27",
        "r42:5x27,0x10,5x27",
        "r43:5x27,0x10,5x27",
        "r44:5x64",
        "r45:5x64",
        "r46:5x64",
        "r47:5x64",
        "r48:5x64",
        "r49:5x64",
        "r50:5x64",
        "r51:5x5x64",
        "r52:5x64",
        "r53:5x64",
        "r54:5x64",
        "r55:5x64",
        "r56:5x64",
        "r57:5x64",
        "r58:5x64",
        "r59:5x64",
        "r60:5x64",
        "r61:5x64",
        "r62:5x64",
        "r63:4x64"
    ]
    win_grid = np.zeros((64, 64), dtype=int)
    for i, row_str in enumerate(win_rows):
        if row_str.startswith("r"):
            parts = row_str.split(":")[1].split(",")
            col = 0
            for part in parts:
                val, count = part.split("x")
                val = int(val)
                count = int(count)
                win_grid[i, col:col+count] = val
                col += count
    return win_grid

win_state_grid = get_win_state()