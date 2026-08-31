import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    H, W = g.shape
    if action == 6 and data is not None:
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        r, c = y, x
        # progress bar: fill next 2 cells of bottom row, right-to-left, wrapping
        filled = int(np.sum(g[-1, :] == 5))
        if filled < W:
            col = W - 2 - (filled // 2) * 2
            if col >= 0:
                g[-1, col:col + 2] = 5
        # move the red (15) 3x3 block to the clicked center
        if 1 <= r <= H - 2 and 1 <= c <= W - 2:
            block = g[r - 1:r + 2, c - 1:c + 2]
            if np.all(block == 15):
                # find existing red 3x3 block and clear it
                found = False
                for rr in range(H):
                    for cc in range(W):
                        if g[rr, cc] == 15:
                            # check 3x3 all red
                            if 1 <= rr <= H - 2 and 1 <= cc <= W - 2:
                                if np.all(g[rr - 1:rr + 2, cc - 1:cc + 2] == 15):
                                    g[rr - 1:rr + 2, cc - 1:cc + 2] = 5
                                    found = True
                                    break
                    if found:
                        break
    return g

def is_level_complete(grid):
    g = np.array(grid)
    # bottom row fully filled with 5
    return bool(np.all(g[-1, :] == 5))
