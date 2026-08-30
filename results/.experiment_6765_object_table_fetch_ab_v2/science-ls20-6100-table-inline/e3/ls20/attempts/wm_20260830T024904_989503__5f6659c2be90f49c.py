import numpy as np

def _find_obj(g):
    # 12 block is top 2 rows of the 5x5 object; extend down 3 rows for the 9 part
    ys, xs = np.where(g == 12)
    if len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    if (y1 - y0 + 1) != 2 or (x1 - x0 + 1) != 5:
        return None
    # full object: rows y0..y0+4, cols x0..x1
    fy1 = y0 + 4
    if fy1 >= g.shape[0]:
        return None
    return (y0, x0, fy1, x1)

def _move(g, y0, x0, y1, x1, dy, dx):
    block = g[y0:y1+1, x0:x1+1].copy()
    ny0, nx0 = y0 + dy, x0 + dx
    ny1, nx1 = y1 + dy, x1 + dx
    if ny0 < 0 or nx0 < 0 or ny1 >= g.shape[0] or nx1 >= g.shape[1]:
        return g
    g[y0:y1+1, x0:x1+1] = 3
    g[ny0:ny1+1, nx0:nx1+1] = block
    return g

def engine(grid, action, data):
    g = np.array(grid, dtype=int)
    obj = _find_obj(g)
    if obj is None:
        return g
    y0, x0, y1, x1 = obj
    if action == 3:
        g = _move(g, y0, x0, y1, x1, 0, -5)
    elif action == 1:
        g = _move(g, y0, x0, y1, x1, -5, 0)
    elif action == 2:
        g = _move(g, y0, x0, y1, x1, 0, 5)
    elif action == 4:
        g = _move(g, y0, x0, y1, x1, 5, 0)
    return g

def is_level_complete(grid):
    return False
