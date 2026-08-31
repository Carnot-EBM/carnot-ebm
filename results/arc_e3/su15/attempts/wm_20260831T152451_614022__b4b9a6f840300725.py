import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    H, W = g.shape
    if action == 6 and data is not None:
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        r, c = y, x
        if 0 <= r < H and 0 <= c < W and g[r, c] == 10:
            # 2x2 block of color 6 with clicked cell as bottom-right corner
            block = set()
            for dr in (-1, 0):
                for dc in (-1, 0):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        g[rr, cc] = 6
                        block.add((rr, cc))
            # erase (->5) any other color-10 stars within Chebyshev radius 2 of the click
            consumed = 1  # the clicked star
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in block and g[rr, cc] == 10:
                        g[rr, cc] = 5
                        consumed += 1
            # progress bar: fill `consumed` cells from the right of the last row, 0 -> 5
            if H > 0:
                last = H - 1
                for cc in range(W - 1, -1, -1):
                    if consumed <= 0:
                        break
                    if g[last, cc] == 0:
                        g[last, cc] = 5
                        consumed -= 1
    return g

def is_level_complete(grid):
    return False
