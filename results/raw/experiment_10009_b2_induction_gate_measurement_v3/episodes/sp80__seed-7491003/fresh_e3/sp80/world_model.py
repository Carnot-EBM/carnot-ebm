import numpy as np


def _build_initial_grid():
    grid = np.zeros((64, 64), dtype=int)
    # Row 0: all 14
    grid[0, :] = 14
    # Rows 1-3: 12x36, 4x4, 12x24
    for r in range(1, 4):
        grid[r, :36] = 12
        grid[r, 36:40] = 4
        grid[r, 40:] = 12
    # Rows 4-7: 12x36, 6x4, 12x24
    for r in range(4, 8):
        grid[r, :36] = 12
        grid[r, 36:40] = 6
        grid[r, 40:] = 12
    # Rows 8-15: all 12
    for r in range(8, 16):
        grid[r, :] = 12
    # Rows 16-19: 12x12, 9x20, 12x32
    for r in range(16, 20):
        grid[r, :12] = 12
        grid[r, 12:32] = 9
        grid[r, 32:] = 12
    # Rows 20-51: all 12
    for r in range(20, 52):
        grid[r, :] = 12
    # Rows 52-55: 12x16, 11x4, 12x4, 11x4, 12x12, 11x4, 12x4, 11x4, 12x12
    for r in range(52, 56):
        c = 0
        runs = [(12, 16), (11, 4), (12, 4), (11, 4), (12, 12), (11, 4), (12, 4), (11, 4), (12, 12)]
        for v, n in runs:
            grid[r, c:c+n] = v
            c += n
    # Rows 56-59: 12x16, 11x12, 12x12, 11x12, 12x12
    for r in range(56, 60):
        c = 0
        runs = [(12, 16), (11, 12), (12, 12), (11, 12), (12, 12)]
        for v, n in runs:
            grid[r, c:c+n] = v
            c += n
    # Rows 60-63: all 1
    for r in range(60, 64):
        grid[r, :] = 1
    return grid


def engine(grid, action, data):
    g = grid.copy()
    h, w = g.shape

    if action == 6:
        px, py = data['x'], data['y']
        lx, ly = px // 1, py // 1
        if 0 <= ly < h and 0 <= lx < w:
            g[ly, lx] = 0
        return g

    # Track the "active" block position. The active block is a 4-row x 20-col region of color 9.
    # Find it by scanning for contiguous 9 regions.
    def find_active_block(g):
        mask = (g == 9)
        rows_with_9 = np.where(mask.any(axis=1))[0]
        if len(rows_with_9) == 0:
            return None
        cols_with_9 = np.where(mask.any(axis=0))[0]
        if len(cols_with_9) == 0:
            return None
        y0, y1 = rows_with_9.min(), rows_with_9.max()
        x0, x1 = cols_with_9.min(), cols_with_9.max()
        return (int(y0), int(x0), int(y1), int(x1))

    blk = find_active_block(g)
    if blk is None:
        return g

    y0, x0, y1, x1 = blk
    bh = y1 - y0 + 1
    bw = x1 - x0 + 1

    dy, dx = 0, 0
    if action == 1:
        dy, dx = -1, 0
    elif action == 2:
        dy, dx = 1, 0
    elif action == 3:
        dy, dx = 0, -1
    elif action == 4:
        dy, dx = 0, 1
    elif action == 5:
        dy, dx = -1, 0

    ny0 = y0 + dy
    nx0 = x0 + dx
    ny1 = y1 + dy
    nx1 = x1 + dx

    # Check bounds
    if ny0 < 0 or ny1 >= h or nx0 < 0 or nx1 >= w:
        return g

    # Check collision with non-12 cells in the destination region (excluding current block area)
    dest_region = g[ny0:ny1+1, nx0:nx1+1]
    cur_region = g[y0:y1+1, x0:x1+1]

    # The destination should be all 12s (background) for movement to proceed
    # But we need to check if any cell in dest is not 12 AND not part of the current block
    blocked = False
    for r in range(ny0, ny1+1):
        for c in range(nx0, nx1+1):
            val = g[r, c]
            if val != 12 and not (y0 <= r <= y1 and x0 <= c <= x1):
                blocked = True
                break
        if blocked:
            break

    if blocked:
        return g

    # Move the block: clear old position, set new position
    g[y0:y1+1, x0:x1+1] = 12
    g[ny0:ny1+1, nx0:nx1+1] = 9

    return g


def is_level_complete(grid):
    # Check if there are no color-9 cells remaining (block collected/removed)
    # Or some other win condition based on observed data
    # From observations, the game seems to involve moving a 4x20 block around.
    # Win state likely when block reaches a specific target or all blocks are cleared.
    # Since we don't have an explicit win state grid, use heuristic:
    # The level is complete when the active 9-block has been moved off-grid or removed.
    mask_9 = (grid == 9)
    if not mask_9.any():
        return True
    return False