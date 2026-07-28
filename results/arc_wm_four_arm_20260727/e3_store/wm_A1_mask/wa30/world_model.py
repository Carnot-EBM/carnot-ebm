import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 3:
        if data is not None:
            px, py = data['x'], data['y']
            r, c = py // 1, px // 1
            if 0 <= r < H and 0 <= c < W:
                grid[r, c] = 5
        return grid
    elif action == 2:
        # Gravity: move all 7s down within their column
        for c in range(W):
            col = grid[:, c]
            new_col = np.zeros(H, dtype=int)
            count_7 = 0
            for r in range(H - 1, -1, -1):
                if col[r] == 7:
                    count_7 += 1
                else:
                    if count_7 > 0:
                        new_col[r - count_7 + 1] = 7
                        count_7 -= 1
                    new_col[r] = col[r]
            grid[:, c] = new_col
        return grid
    return grid

def is_level_complete(grid):
    # Check if all 7s have reached the bottom (row 63)
    H, W = grid.shape
    for c in range(W):
        col = grid[:, c]
        # Find the last 7 in the column
        last_7_idx = -1
        for r in range(H - 1, -1, -1):
            if col[r] == 7:
                last_7_idx = r
                break
        if last_7_idx != 63:
            return False
    return True