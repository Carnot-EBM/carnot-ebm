import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    # Find the 9-colored "player" object: the largest connected component of 9s
    # (4-connectivity). Move it according to the action.
    # Actions: 1=up, 2=down, 3=left, 4=right, 5=?, 6=click, 7=?
    # We'll implement a generic "move the 9 blob by 1 in the action direction,
    # leaving a trail of 2s behind, and the blob becomes 9 where it lands."
    # This is a rough first guess; refine from mismatch report.

    def label9():
        seen = np.zeros((H, W), dtype=bool)
        best = None
        for i in range(H):
            for j in range(W):
                if g[i, j] == 9 and not seen[i, j]:
                    stack = [(i, j)]
                    comp = []
                    seen[i, j] = True
                    while stack:
                        r, c = stack.pop()
                        comp.append((r, c))
                        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < H and 0 <= nc < W and not seen[nr,nc] and g[nr,nc]==9:
                                seen[nr,nc]=True
                                stack.append((nr,nc))
                    if best is None or len(comp) > len(best):
                        best = comp
        return best

    comp = label9()
    if comp is None:
        return g

    # direction for action
    dirs = {1:(-1,0), 2:(1,0), 3:(0,-1), 4:(0,1)}
    d = dirs.get(action)
    if d is None:
        return g
    dr, dc = d
    newg = g.copy()
    # clear old 9s
    for (r,c) in comp:
        newg[r,c] = 5  # becomes floor
    # place new 9s shifted
    for (r,c) in comp:
        nr, nc = r+dr, c+dc
        if 0 <= nr < H and 0 <= nc < W:
            newg[nr,nc] = 9
    return newg

def is_level_complete(grid):
    return False
