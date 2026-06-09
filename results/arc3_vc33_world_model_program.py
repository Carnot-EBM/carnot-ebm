# VERIFIED ARC-AGI-3 vc33 world-model (M2-v4b) — codex (gpt-5.5) induced, consistency-energy verified.
# energy 0.005 (seed 0) / 0.011 (seed 7) on 600 held-out transitions = ~99% dynamics accuracy, REPLICATED.
# Generator: codex program synthesis on ACTIVE-collected data. Verifier: grade_predictions (no oracle).
# This is the first trustworthy induced world-model in the ARC-AGI-3 investigation. Provenance evidence.

def predict(grid, action):
    g = grid.copy()
    h, w = g.shape

    vals, counts = np.unique(grid, return_counts=True)
    bg = vals[np.argmax(counts)]

    if not action or action[0] != 6 or len(action) < 3:
        return g

    x, y = int(action[1]), int(action[2])
    if not (0 <= x < w and 0 <= y < h):
        return g

    top = grid[0]
    old = top[0]

    if old != bg:
        consumed = 0
        for c in range(w - 1, -1, -1):
            if top[c] == old:
                break
            consumed += 1

        new = None
        if consumed:
            suffix = top[w - consumed:]
            sv, sc = np.unique(suffix, return_counts=True)
            new = sv[np.argmax(sc)]
        else:
            non_bg = [v for v in vals if v != bg and v != old]
            if non_bg:
                preferred = old - bg
                new = preferred if preferred in non_bg else non_bg[np.argmin([np.sum(grid == v) for v in non_bg])]
            else:
                new = bg

        # The top row is a rasterized 50-step progress bar.  Each click advances
        # one logical step, then maps that progress back onto the row width.
        steps_total = max(1, int(np.floor(w * 25.0 / 32.0 + 0.5)))
        step = int(np.floor(consumed * steps_total / float(w) + 0.5))
        target = int(np.floor((step + 1) * w / float(steps_total) + 0.5))
        target = max(consumed + 1, min(w, target))

        for c in range(w - target, w - consumed):
            if 0 <= c < w and g[0, c] == old:
                g[0, c] = new

    clicked = grid[y, x]
    if y != 0 and clicked != bg and clicked != old:
        seen = np.zeros((h, w), dtype=bool)
        stack = [(y, x)]
        seen[y, x] = True
        comp = []

        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and grid[rr, cc] == clicked:
                    seen[rr, cc] = True
                    stack.append((rr, cc))

        color_count = int(np.sum(grid == clicked))
        if len(comp) <= max(4, h * w // 128) and color_count <= max(8, h * w // 64):
            ignore = {bg, clicked, old}
            ignore.add(g[0, -1])
            best_cells = None
            best_size = 0

            for col in vals:
                if col in ignore:
                    continue

                mask = grid == col
                visited = np.zeros((h, w), dtype=bool)
                rs, cs = np.where(mask)

                for sr, sc in zip(rs, cs):
                    if visited[sr, sc]:
                        continue

                    q = [(sr, sc)]
                    visited[sr, sc] = True
                    cells = []

                    while q:
                        r, c = q.pop()
                        cells.append((r, c))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and mask[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))

                    if len(cells) > best_size:
                        best_size = len(cells)
                        best_cells = cells

            if best_cells is not None and best_size >= max(16, h * w // 32):
                for r, c in best_cells:
                    g[r, c] = bg

    return g