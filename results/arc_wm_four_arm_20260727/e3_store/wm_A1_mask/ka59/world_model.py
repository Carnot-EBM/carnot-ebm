import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            if grid[py, px] == 3:
                new_grid[py, px] = 0
    elif action == 4:
        # Action 4: Move right (implied by context of pushing blocks)
        # Identify blocks of color 15 and push them right
        for r in range(H):
            col = 0
            while col < W:
                if new_grid[r, col] == 15:
                    # Find the end of this block
                    end_col = col
                    while end_col < W and new_grid[r, end_col] == 15:
                        end_col += 1
                    # Push the block to the right
                    for c in range(col, end_col):
                        new_grid[r, c] = 0
                    for c in range(end_col, W):
                        if new_grid[r, c] == 15:
                            new_grid[r, c] = 0
                        elif new_grid[r, c] == 10:
                            new_grid[r, c] = 15
                            break
                else:
                    col += 1
    elif action == 1:
        # Action 1: Move left (implied by context)
        # Identify blocks of color 15 and push them left
        for r in range(H):
            col = W - 1
            while col >= 0:
                if new_grid[r, col] == 15:
                    # Find the start of this block
                    start_col = col
                    while start_col >= 0 and new_grid[r, start_col] == 15:
                        start_col -= 1
                    start_col += 1
                    # Push the block to the left
                    for c in range(start_col, col + 1):
                        new_grid[r, c] = 0
                    for c in range(start_col - 1, -1, -1):
                        if new_grid[r, c] == 15:
                            new_grid[r, c] = 0
                        elif new_grid[r, c] == 10:
                            new_grid[r, c] = 15
                            break
                else:
                    col -= 1
    elif action == 2:
        # Action 2: Move up
        for c in range(W):
            row = H - 1
            while row >= 0:
                if new_grid[row, c] == 15:
                    # Find the start of this block
                    start_row = row
                    while start_row >= 0 and new_grid[start_row, c] == 15:
                        start_row -= 1
                    start_row += 1
                    # Push the block up
                    for r in range(start_row, row + 1):
                        new_grid[r, c] = 0
                    for r in range(start_row - 1, -1, -1):
                        if new_grid[r, c] == 15:
                            new_grid[r, c] = 0
                        elif new_grid[r, c] == 10:
                            new_grid[r, c] = 15
                            break
                else:
                    row -= 1
    elif action == 3:
        # Action 3: Move down
        for c in range(W):
            row = 0
            while row < H:
                if new_grid[row, c] == 15:
                    # Find the end of this block
                    end_row = row
                    while end_row < H and new_grid[end_row, c] == 15:
                        end_row += 1
                    # Push the block down
                    for r in range(row, end_row):
                        new_grid[r, c] = 0
                    for r in range(end_row, H):
                        if new_grid[r, c] == 15:
                            new_grid[r, c] = 0
                        elif new_grid[r, c] == 10:
                            new_grid[r, c] = 15
                            break
                else:
                    row += 1
    elif action == 5:
        # Action 5: Move diagonal (implied by context)
        # This is a simplified version; actual diagonal movement might be more complex
        pass
    
    return new_grid

def is_level_complete(grid):
    # Check if the level is complete
    # This is a simplified version; actual completion criteria might be more complex
    return np.all(grid == 0)