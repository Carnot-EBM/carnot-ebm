def engine(grid, action, data):
    h, w = grid.shape
    if action == 6:
        if data is None:
            return grid
        px, py = data["x"], data["y"]
        if px < 0 or px >= w or py < 0 or py >= h:
            return grid
        new_grid = grid.copy()
        new_grid[py, px] = 5
        return new_grid
    return grid


def is_level_complete(grid):
    h, w = grid.shape
    if h != 64 or w != 64:
        return False
    for r in range(h):
        row = grid[r, :]
        if row[0] != 14:
            return False
        if row[1] != 3:
            return False
        for c in range(2, w):
            if row[c] != 10:
                return False
    return True
