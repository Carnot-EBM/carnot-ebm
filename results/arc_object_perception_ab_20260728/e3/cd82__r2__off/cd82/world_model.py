import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if grid[r, c] != 5:
                    new_grid[r, c] = grid[r, c]
                    break
            else:
                for r in range(H - 1, -1, -1):
                    if grid[r, c] != 5:
                        new_grid[r, c] = grid[r, c]
                        break
                else:
                    new_grid[0, c] = 5
        return new_grid
    
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H):
                if grid[r, c] != 5:
                    new_grid[r, c] = grid[r, c]
                    break
            else:
                for r in range(H):
                    if grid[r, c] != 5:
                        new_grid[r, c] = grid[r, c]
                        break
                else:
                    new_grid[H - 1, c] = 5
        return new_grid
    
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if grid[r, c] != 5:
                    new_grid[r, c] = grid[r, c]
                    break
            else:
                for c in range(W - 1, -1, -1):
                    if grid[r, c] != 5:
                        new_grid[r, c] = grid[r, c]
                        break
                else:
                    new_grid[r, 0] = 5
        return new_grid
    
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 5:
                    new_grid[r, c] = grid[r, c]
                    break
            else:
                for c in range(W):
                    if grid[r, c] != 5:
                        new_grid[r, c] = grid[r, c]
                        break
                else:
                    new_grid[r, W - 1] = 5
        return new_grid
    
    elif action == 5:
        # Move Up-Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 5:
                    new_grid[r, c] = grid[r, c]
                    break
            else:
                for c in range(W):
                    if grid[r, c] != 5:
                        new_grid[r, c] = grid[r, c]
                        break
                else:
                    new_grid[r, W - 1] = 5
        return new_grid
    
    elif action == 6:
        # Click
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        if 0 <= logical_y < H and 0 <= logical_x < W:
            new_grid[logical_y, logical_x] = 5
        return new_grid
    
    elif action == 7:
        # Move Down-Left
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 5:
                    new_grid[r, c] = grid[r, c]
                    break
            else:
                for c in range(W):
                    if grid[r, c] != 5:
                        new_grid[r, c] = grid[r, c]
                        break
                else:
                    new_grid[r, 0] = 5
        return new_grid
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if the grid matches the win state pattern
    # The win state has specific patterns in the first few rows
    # We check for the presence of the win state pattern
    # This is a simplified check based on the win state provided
    
    # Check row 0
    r0 = grid[0]
    if not (np.sum(r0 == 5) == 16 and np.sum(r0 == 4) == 2 and np.sum(r0 == 3) == 46):
        return False
    
    # Check row 1
    r1 = grid[1]
    if not (np.sum(r1 == 5) == 16 and np.sum(r1 == 4) == 2 and np.sum(r1 == 3) == 46):
        return False
    
    # Check row 2
    r2 = grid[2]
    if not (np.sum(r2 == 5) == 16 and np.sum(r2 == 4) == 2 and np.sum(r2 == 3) == 14 and np.sum(r2 == 4) == 5 and np.sum(r2 == 3) == 1 and np.sum(r2 == 4) == 5 and np.sum(r2 == 3) == 1 and np.sum(r2 == 4) == 5 and np.sum(r2 == 3) == 15):
        return False
    
    # Check row 3
    r3 = grid[3]
    if not (np.sum(r3 == 5) == 3 and np.sum(r3 == 15) == 9 and np.sum(r3 == 12) == 1 and np.sum(r3 == 4) == 2 and np.sum(r3 == 3) == 14 and np.sum(r3 == 4) == 1 and np.sum(r3 == 0) == 3 and np.sum(r3 == 4) == 1 and np.sum(r3 == 3) == 1 and np.sum(r3 == 4) == 1 and np.sum(r3 == 15) == 3 and np.sum(r3 == 4) == 1 and np.sum(r3 == 3) == 1 and np.sum(r3 == 4) == 1 and np.sum(r3 == 12) == 3 and np.sum(r3 == 4) == 1 and np.sum(r3 == 3) == 15):
        return False
    
    # Check row 4
    r4 = grid[4]
    if not (np.sum(r4 == 5) == 3 and np.sum(r4 == 15) == 8 and np.sum(r4 == 12) == 2 and np.sum(r4 == 4) == 2 and np.sum(r4 == 3) == 14 and np.sum(r4 == 4) == 1 and np.sum(r4 == 0) == 3 and np.sum(r4 == 4) == 1 and np.sum(r4 == 3) == 1 and np.sum(r4 == 4) == 1 and np.sum(r4 == 15) == 3 and np.sum(r4 == 4) == 1 and np.sum(r4 == 3) == 1 and np.sum(r4 == 4) == 1 and np.sum(r4 == 12) == 3 and np.sum(r4 == 4) == 1 and np.sum(r4 == 3) == 15):
        return False
    
    # Check row 5
    r5 = grid[5]
    if not (np.sum(r5 == 5) == 3 and np.sum(r5 == 15) == 7 and np.sum(r5 == 12) == 3 and np.sum(r5 == 4) == 2 and np.sum(r5 == 3) == 14 and np.sum(r5 == 4) == 1 and np.sum(r5 == 0) == 3 and np.sum(r5 == 4) == 1 and np.sum(r5 == 3) == 1 and np.sum(r5 == 4) == 1 and np.sum(r5 == 15) == 3 and np.sum(r5 == 4) == 1 and np.sum(r5 == 3) == 1 and np.sum(r5 == 4) == 1 and np.sum(r5 == 12) == 3 and np.sum(r5 == 4) == 1 and np.sum(r5 == 3) == 15):
        return False
    
    # Check row 6
    r6 = grid[6]
    if not (np.sum(r6 == 5) == 3 and np.sum(r6 == 15) == 6 and np.sum(r6 == 12) == 4 and np.sum(r6 == 4) == 2 and np.sum(r6 == 3) == 14 and np.sum(r6 == 4) == 5 and np.sum(r6 == 3) == 1 and np.sum(r6 == 4) == 5 and np.sum(r6 == 3) == 1 and np.sum(r6 == 4) == 5 and np.sum(r6 == 3) == 15):
        return False
    
    # Check row 7
    r7 = grid[7]
    if not (np.sum(r7 == 5) == 3 and np.sum(r7 == 15) == 5 and np.sum(r7 == 12) == 5 and np.sum(r7 == 4) == 2 and np.sum(r7 == 3) == 20 and np.sum(r7 == 0) == 5 and np.sum(r7 == 3) == 21):
        return False
    
    # Check row 8
    r8 = grid[8]
    if not (np.sum(r8 == 5) == 3 and np.sum(r8 == 0) == 4 and np.sum(r8 == 12) == 6 and np.sum(r8 == 4) == 2 and np.sum(r8 == 3) == 46):
        return False
    
    # Check row 9
    r9 = grid[9]
    if not (np.sum(r9 == 5) == 3 and np.sum(r9 == 0) == 3 and np.sum(r9 == 12) == 7 and np.sum(r9 == 4) == 2 and np.sum(r9 == 5) == 46):
        return False
    
    # Check row 10
    r10 = grid[10]
    if not (np.sum(r10 == 5) == 3 and np.sum(r10 == 0) == 2 and np.sum(r10 == 12) == 8 and np.sum(r10 == 4) == 2 and np.sum(r10 == 5) == 46):
        return False
    
    # Check row 11
    r11 = grid[11]
    if not (np.sum(r11 == 5) == 3 and np.sum(r11 == 0) == 1 and np.sum(r11 == 12) == 9 and np.sum(r11 == 4) == 2 and np.sum(r11 == 5) == 46):
        return False
    
    # Check row 12
    r12 = grid[12]
    if not (np.sum(r12 == 5) == 3 and np.sum(r12 == 12) == 10 and np.sum(r12 == 4) == 2 and np.sum(r12 == 5) == 46):
        return False
    
    # Check row 13
    r13 = grid[13]
    if not (np.sum(r13 == 5) == 16 and np.sum(r13 == 4) == 2 and np.sum(r13 == 5) == 46):
        return False
    
    # Check row 14
    r14 = grid[14]
    if not (np.sum(r14 == 5) == 16 and np.sum(r14 == 4) == 2 and np.sum(r14 == 5) == 46):
        return False
    
    # Check row 15
    r15 = grid[15]
    if not (np.sum(r15 == 5) == 16 and np.sum(r15 == 4) == 2 and np.sum(r15 == 5) == 46):
        return False
    
    # Check row 16
    r16 = grid[16]
    if not (np.sum(r16 == 4) == 18 and np.sum(r16 == 5) == 46):
        return False
    
    # Check row 17
    r17 = grid[17]
    if not (np.sum(r17 == 4) == 18 and np.sum(r17 == 5) == 46):
        return False
    
    # Check row 18
    r18 = grid[18]
    if not (np.sum(r18 == 5) == 64):
        return False
    
    # Check row 19
    r19 = grid[19]
    if not (np.sum(r19 == 5) == 64):
        return False
    
    # Check row 20
    r20 = grid[20]
    if not (np.sum(r20 == 5) == 64):
        return False
    
    # Check row 21
    r21 = grid[21]
    if not (np.sum(r21 == 5) == 64):
        return False
    
    # Check row 22
    r22 = grid[22]
    if not (np.sum(r22 == 5) == 64):
        return False
    
    # Check row 23
    r23 = grid[23]
    if not (np.sum(r23 == 5) == 64):
        return False
    
    # Check row 24
    r24 = grid[24]
    if not (np.sum(r24 == 5) == 25 and np.sum(r24 == 2) == 14 and np.sum(r24 == 5) == 25):
        return False
    
    # Check row 25
    r25 = grid[25]
    if not (np.sum(r25 == 5) == 25 and np.sum(r25 == 2) == 1 and np.sum(r25 == 15) == 12 and np.sum(r25 == 2) == 1 and np.sum(r25 == 5) == 25):
        return False
    
    # Check row 26
    r26 = grid[26]
    if not (np.sum(r26 == 5) == 25 and np.sum(r26 == 2) == 1 and np.sum(r26 == 15) == 12 and np.sum(r26 == 2) == 1 and np.sum(r26 == 5) == 25):
        return False
    
    # Check row 27
    r27 = grid[27]
    if not (np.sum(r27 == 5) == 25 and np.sum(r27 == 2) == 1 and np.sum(r27 == 15) == 12 and np.sum(r27 == 2) == 1 and np.sum(r27 == 5) == 25):
        return False
    
    # Check row 28
    r28 = grid[28]
    if not (np.sum(r28 == 5) == 25 and np.sum(r28 == 2) == 1 and np.sum(r28 == 15) == 12 and np.sum(r28 == 2) == 1 and np.sum(r28 == 5) == 25):
        return False
    
    # Check row 29
    r29 = grid[29]
    if not (np.sum(r29 == 5) == 25 and np.sum(r29 == 2) == 1 and np.sum(r29 == 15) == 12 and np.sum(r29 == 2) == 1 and np.sum(r29 == 5) == 25):
        return False
    
    # Check row 30
    r30 = grid[30]
    if not (np.sum(r30 == 5) == 25 and np.sum(r30 == 2) == 1 and np.sum(r30 == 15) == 12 and np.sum(r30 == 2) == 1 and np.sum(r30 == 5) == 25):
        return False
    
    # Check row 31
    r31 = grid[31]
    if not (np.sum(r31 == 5) == 25 and np.sum(r31 == 2) == 1 and np.sum(r31 == 15) == 12 and np.sum(r31 == 2) == 1 and np.sum(r31 == 5) == 25):
        return False
    
    # Check row 32
    r32 = grid[32]
    if not (np.sum(r32 == 5) == 25 and np.sum(r32 == 2) == 1 and np.sum(r32 == 5) == 12 and np.sum(r32 == 2) == 1 and np.sum(r32 == 5) == 25):
        return False
    
    # Check row 33
    r33 = grid[33]
    if not (np.sum(r33 == 5) == 64):
        return False