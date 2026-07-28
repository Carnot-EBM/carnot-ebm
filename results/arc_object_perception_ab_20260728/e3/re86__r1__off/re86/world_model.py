import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] == 5:
                    if r > 0 and grid[r - 1, c] != 5:
                        new_grid[r, c] = 0
                        new_grid[r - 1, c] = 5
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H):
                if grid[r, c] == 5:
                    if r < H - 1 and grid[r + 1, c] != 5:
                        new_grid[r, c] = 0
                        new_grid[r + 1, c] = 5
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if grid[r, c] == 5:
                    if c > 0 and grid[r, c - 1] != 5:
                        new_grid[r, c] = 0
                        new_grid[r, c - 1] = 5
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    if c < W - 1 and grid[r, c + 1] != 5:
                        new_grid[r, c] = 0
                        new_grid[r, c + 1] = 5
    elif action == 5:
        # Toggle 0 <-> 9
        mask = (grid == 0) | (grid == 9)
        new_grid[mask] = 9 if grid[mask] == 0 else 0
    elif action == 6:
        # Click (data={'x':px, 'y':py})
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            if 0 <= py < H and 0 <= px < W:
                val = grid[py, px]
                if val == 0:
                    new_grid[py, px] = 9
                elif val == 9:
                    new_grid[py, px] = 0
    elif action == 7:
        # Toggle 0 <-> 15
        mask = (grid == 0) | (grid == 15)
        new_grid[mask] = 15 if grid[mask] == 0 else 0
        
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    target = np.zeros((H, W), dtype=int)
    target[0] = 5
    target[1] = 5
    target[2] = np.array([5]*20 + [4]*3 + [5]*41)
    target[3] = np.array([5]*20 + [4]*1 + [13]*1 + [4]*1 + [5]*41)
    target[4] = np.array([5]*20 + [4]*3 + [5]*41)
    target[5] = np.array([5]*64)
    target[6] = np.array([5]*64)
    target[7] = np.array([5]*16 + [12]*1 + [5]*21 + [12]*1 + [5]*25)
    target[8] = np.array([5]*17 + [12]*1 + [5]*8 + [4]*3 + [5]*8 + [12]*1 + [5]*26)
    target[9] = np.array([5]*18 + [12]*1 + [5]*7 + [4]*1 + [13]*1 + [4]*1 + [5]*7 + [12]*1 + [5]*27)
    target[10] = np.array([5]*19 + [12]*1 + [5]*6 + [4]*3 + [5]*6 + [12]*1 + [5]*28)
    target[11] = np.array([5]*11 + [4]*3 + [5]*6 + [12]*1 + [5]*13 + [12]*1 + [5]*29)
    target[12] = np.array([5]*11 + [4]*1 + [13]*1 + [4]*1 + [5]*7 + [12]*1 + [5]*11 + [12]*1 + [5]*30)
    target[13] = np.array([5]*11 + [4]*3 + [5]*8 + [12]*1 + [5]*9 + [12]*1 + [5]*31)
    target[14] = np.array([5]*23 + [12]*1 + [5]*7 + [12]*1 + [5]*32)
    target[15] = np.array([5]*24 + [12]*1 + [5]*5 + [12]*1 + [5]*33)
    target[16] = np.array([5]*25 + [12]*1 + [5]*3 + [12]*1 + [5]*34)
    target[17] = np.array([5]*26 + [12]*1 + [5]*1 + [12]*1 + [5]*35)
    target[18] = np.array([5]*27 + [0]*1 + [5]*36)
    target[19] = np.array([5]*26 + [12]*1 + [5]*1 + [12]*1 + [5]*35)
    target[20] = np.array([5]*25 + [12]*1 + [5]*3 + [12]*1 + [5]*34)
    target[21] = np.array([5]*24 + [12]*1 + [5]*5 + [12]*1 + [5]*8 + [13]*1 + [5]*24)
    target[22] = np.array([5]*23 + [12]*1 + [5]*7 + [12]*1 + [5]*6 + [13]*1 + [5]*1 + [13]*1 + [5]*23)
    target[23] = np.array([5]*22 + [12]*1 + [5]*9 + [12]*1 + [5]*4 + [13]*1 + [5]*3 + [13]*1 + [5]*22)
    target[24] = np.array([5]*21 + [12]*1 + [5]*11 + [12]*1 + [5]*2 + [13]*1 + [5]*5 + [13]*1 + [5]*21)
    target[25] = np.array([5]*20 + [12]*1 + [5]*13 + [12]*1 + [13]*1 + [5]*7 + [13]*1 + [5]*20)
    target[26] = np.array([5]*19 + [12]*1 + [5]*14 + [13]*1 + [12]*1 + [5]*8 + [13]*1 + [5]*19)
    target[27] = np.array([5]*18 + [12]*1 + [5]*14 + [13]*1 + [5]*2 + [12]*1 + [5]*8 + [13]*1 + [5]*18)
    target[28] = np.array([5]*17 + [12]*1 + [5]*14 + [13]*1 + [5]*4 + [12]*1 + [5]*8 + [13]*1 + [5]*17)
    target[29] = np.array([5]*16 + [12]*1 + [5]*14 + [13]*1 + [5]*6 + [12]*1 + [5]*8 + [13]*1 + [9]*1 + [5]*15)
    target[30] = np.array([5]*30 + [13]*1 + [5]*17 + [9]*1 + [5]*15)
    target[31] = np.array([5]*31 + [13]*1 + [5]*15 + [13]*1 + [9]*1 + [5]*15)
    target[32] = np.array([5]*32 + [13]*1 + [5]*13 + [13]*1 + [5]*1 + [9]*1 + [5]*15)
    target[33] = np.array([5]*33 + [13]*1 + [5]*11 + [13]*1 + [5]*2 + [9]*1 + [5]*15)
    target[34] = np.array([5]*34 + [13]*1 + [5]*9 + [13]*1 + [5]*3 + [9]*1 + [5]*15)
    target[35] = np.array([5]*26 + [4]*3 + [5]*6 + [13]*1 + [5]*7 + [13]*1 + [5]*4 + [9]*1 + [5]*15)
    target[36] = np.array([5]*26 + [4]*1 + [9]*1 + [4]*1 + [5]*7 + [13]*1 + [5]*5 + [13]*1 + [5]*5 + [9]*1 + [5]*15)
    target[37] = np.array([5]*26 + [4]*3 + [5]*8 + [13]*1 + [5]*3 + [13]*1 + [5]*6 + [9]*1 + [5]*15)
    target[38] = np.array([5]*8 + [4]*3 + [5]*27 + [13]*1 + [5]*1 + [13]*1 + [5]*7 + [9]*1 + [5]*15)
    target[39] = np.array([5]*8 + [4]*1 + [12]*1 + [4]*1 + [5]*28 + [13]*1 + [5]*8 + [9]*1 + [5]*15)
    target[40] = np.array([5]*8 + [4]*3 + [5]*37 + [9]*1 + [5]*15)
    target[41] = np.array([5]*23 + [4]*3 + [5]*22 + [9]*1 + [5]*15)
    target[42] = np.array([5]*23 + [4]*1 + [12]*1 + [4]*1 + [5]*9 + [9]*27 + [5]*2)
    target[43] = np.array([5]*23 + [4]*3 + [5]*22 + [9]*1 + [5]*15)
    target[44] = np.array([5]*48 + [9]*1 + [5]*15)
    target[45] = np.array([5]*48 + [9]*1 + [5]*15)
    target[46] = np.array([5]*48 + [9]*1 + [5]*15)
    target[47] = np.array([5]*14 + [4]*3 + [5]*15 + [4]*3 + [5]*13 + [9]*1 + [5]*15)
    target[48] = np.array([5]*14 + [4]*1 + [9]*1 + [4]*1 + [5]*15 + [4]*1 + [9]*1 + [4]*1 + [5]*13 + [9]*1 + [5]*15)
    target[49] = np.array([5]*14 + [4]*3 + [5]*15 + [4]*3 + [5]*13 + [9]*1 + [5]*15)
    target[50] = np.array([5]*48 + [9]*1 + [5]*15)
    target[51] = np.array([5]*48 + [9]*1 + [5]*15)
    target[52] = np.array([5]*48 + [9]*1 + [5]*15)
    target[53] = np.array([5]*48 + [9]*1 + [5]*15)
    target[54] = np.array([5]*48 + [9]*1 + [5]*15)
    target[55] = np.array([5]*48 + [9]*1 + [5]*15)
    target[56] = np.array([5]*8 + [4]*3 + [5]*53)
    target[57] = np.array([5]*8 + [4]*1 + [12]*1 + [4]*1 + [5]*53)
    target[58] = np.array([5]*8 + [4]*3 + [5]*53)
    target[59] = np.array([5]*26 + [4]*3 + [5]*35)
    target[60] = np.array([5]*26 + [4]*1 + [9]*1 + [4]*1 + [5]*35)
    target[61] = np.array([5]*26 + [4]*3 + [5]*35)
    target[62] = np.array([5]*64)
    target[63] = np.array([15]*64)
    
    return np.array_equal(grid, target)