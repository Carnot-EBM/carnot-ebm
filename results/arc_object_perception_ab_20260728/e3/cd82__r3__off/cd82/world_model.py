import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] != 0 and grid[r - 1, c] == 0:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c] = grid[r, c]
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H - 1):
                if grid[r, c] != 0 and grid[r + 1, c] == 0:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c] = grid[r, c]
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if grid[r, c] != 0 and grid[r, c - 1] == 0:
                    new_grid[r, c] = 0
                    new_grid[r, c - 1] = grid[r, c]
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 0 and grid[r, c + 1] == 0:
                    new_grid[r, c] = 0
                    new_grid[r, c + 1] = grid[r, c]
    elif action == 5:
        # Toggle 0s to 15s
        new_grid = grid.copy()
        new_grid[grid == 0] = 15
    elif action == 6:
        # Click (no-op in this model)
        pass
    elif action == 7:
        # Move Diagonal (Up-Right)
        for r in range(H - 1):
            for c in range(W - 1):
                if grid[r, c] != 0 and grid[r - 1, c + 1] == 0:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c + 1] = grid[r, c]
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if the grid matches the win state pattern
    # The win state has specific patterns in the first few rows
    # We check if the grid matches the expected win state structure
    
    # Check row 0
    r0 = grid[0]
    if not np.array_equal(r0, np.array([5]*16 + [4]*2 + [3]*46)):
        return False
    
    # Check row 1
    r1 = grid[1]
    if not np.array_equal(r1, np.array([5]*16 + [4]*2 + [3]*46)):
        return False
    
    # Check row 2
    r2 = grid[2]
    if not np.array_equal(r2, np.array([5]*16 + [4]*2 + [3]*14 + [4]*5 + [3]*1 + [4]*5 + [3]*1 + [4]*5 + [3]*15)):
        return False
    
    # Check row 3
    r3 = grid[3]
    if not np.array_equal(r3, np.array([5]*3 + [15]*9 + [12]*1 + [5]*3 + [4]*2 + [3]*14 + [4]*1 + [0]*3 + [4]*1 + [3]*1 + [4]*1 + [15]*3 + [4]*1 + [3]*1 + [4]*1 + [12]*3 + [4]*1 + [3]*15)):
        return False
    
    # Check row 4
    r4 = grid[4]
    if not np.array_equal(r4, np.array([5]*3 + [15]*8 + [12]*2 + [5]*3 + [4]*2 + [3]*14 + [4]*1 + [0]*3 + [4]*1 + [3]*1 + [4]*1 + [15]*3 + [4]*1 + [3]*1 + [4]*1 + [12]*3 + [4]*1 + [3]*15)):
        return False
    
    # Check row 5
    r5 = grid[5]
    if not np.array_equal(r5, np.array([5]*3 + [15]*7 + [12]*3 + [5]*3 + [4]*2 + [3]*14 + [4]*1 + [0]*3 + [4]*1 + [3]*1 + [4]*1 + [15]*3 + [4]*1 + [3]*1 + [4]*1 + [12]*3 + [4]*1 + [3]*15)):
        return False
    
    # Check row 6
    r6 = grid[6]
    if not np.array_equal(r6, np.array([5]*3 + [15]*6 + [12]*4 + [5]*3 + [4]*2 + [3]*14 + [4]*5 + [3]*1 + [4]*5 + [3]*1 + [4]*5 + [3]*15)):
        return False
    
    # Check row 7
    r7 = grid[7]
    if not np.array_equal(r7, np.array([5]*3 + [15]*5 + [12]*5 + [5]*3 + [4]*2 + [3]*20 + [0]*5 + [3]*21)):
        return False
    
    # Check row 8
    r8 = grid[8]
    if not np.array_equal(r8, np.array([5]*3 + [0]*4 + [12]*6 + [5]*3 + [4]*2 + [3]*46)):
        return False
    
    # Check row 9
    r9 = grid[9]
    if not np.array_equal(r9, np.array([5]*3 + [0]*3 + [12]*7 + [5]*3 + [4]*2 + [5]*46)):
        return False
    
    # Check row 10
    r10 = grid[10]
    if not np.array_equal(r10, np.array([5]*3 + [0]*2 + [12]*8 + [5]*3 + [4]*2 + [5]*46)):
        return False
    
    # Check row 11
    r11 = grid[11]
    if not np.array_equal(r11, np.array([5]*3 + [0]*1 + [12]*9 + [5]*3 + [4]*2 + [5]*46)):
        return False
    
    # Check row 12
    r12 = grid[12]
    if not np.array_equal(r12, np.array([5]*3 + [12]*10 + [5]*3 + [4]*2 + [5]*46)):
        return False
    
    # Check row 13
    r13 = grid[13]
    if not np.array_equal(r13, np.array([5]*16 + [4]*2 + [5]*46)):
        return False
    
    # Check row 14
    r14 = grid[14]
    if not np.array_equal(r14, np.array([5]*16 + [4]*2 + [5]*46)):
        return False
    
    # Check row 15
    r15 = grid[15]
    if not np.array_equal(r15, np.array([5]*16 + [4]*2 + [5]*46)):
        return False
    
    # Check row 16
    r16 = grid[16]
    if not np.array_equal(r16, np.array([4]*18 + [5]*46)):
        return False
    
    # Check row 17
    r17 = grid[17]
    if not np.array_equal(r17, np.array([4]*18 + [5]*46)):
        return False
    
    # Check row 18-23
    for r in range(18, 24):
        if not np.array_equal(grid[r], np.array([5]*64)):
            return False
    
    # Check row 24
    r24 = grid[24]
    if not np.array_equal(r24, np.array([5]*25 + [2]*14 + [5]*25)):
        return False
    
    # Check rows 25-31
    for r in range(25, 32):
        if not np.array_equal(grid[r], np.array([5]*25 + [2]*1 + [15]*12 + [2]*1 + [5]*25)):
            return False
    
    # Check row 32
    r32 = grid[32]
    if not np.array_equal(r32, np.array([5]*25 + [2]*1 + [5]*12 + [2]*1 + [5]*25)):
        return False
    
    # Check row 33
    r33 = grid[33]
    if not np.array_equal(r33, np.array([5]*64)):
        return False
    
    # Check rows 34-43
    for r in range(34, 44):
        if not np.array_equal(grid[r], np.array([5]*27 + [0]*10 + [5]*27)):
            return False
    
    # Check rows 44-62
    for r in range(44, 63):
        if not np.array_equal(grid[r], np.array([5]*64)):
            return False
    
    # Check row 63
    r63 = grid[63]
    if not np.array_equal(r63, np.array([4]*64)):
        return False
    
    return True