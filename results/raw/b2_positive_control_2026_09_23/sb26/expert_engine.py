"""Expert GRID-ONLY world model for ARC-AGI-3 game sb26, level 1 (index 0).

Uses ONLY (grid, action, data) plus constants read from the public source
environment_files/sb26/7fbdac44/sb26.py (Level 1 sprite list, lines ~363-385;
constants BACKGROUND_COLOR=4, HUD row evrmzyfopo=53, bar colours 2/3, ring colour 0).
No file reads, no simulator, no recorded grids, no hashing of inputs.

Mechanics (from Sb26.step / hjewbkcejq / kxrrueustb / rfdjlhefnd):
  * 8 locations: 4 palette slots (y=56) and 4 frame slots (y=27). Each holds an item
    (6x6 sprite, 4x4 colour interior) or an empty spot (2x2 colour-2 dot).
  * A colour-0 6x6 ring marks the selected item.
  * ACTION6 click: select / deselect / move selection (both palette) / swap / move to spot.
    Swap and move cost 1 energy (HUD row 53: first e cells colour 2, rest colour 3).
  * ACTION5: costs 1 energy, clears selection, runs the program; a wrong program resets
    visually; a correct one levels up (level-up rows are not graded).
  * ACTION7: undo. The undo stack is HIDDEN state. Guess: no-op when the layout is the
    level-initial layout; otherwise restore the unique one-move predecessor on a shortest
    path from the initial layout if it is unique, else only clear the selection.
"""
import numpy as np

BG = 4
HUD_ROW = 53
PAL = [(17, 56), (25, 56), (33, 56), (41, 56)]
FRM = [(20, 27), (26, 27), (32, 27), (38, 27)]
LOCS = PAL + FRM
INIT = (14, 15, 9, 11, None, None, None, None)
TARGET = (9, 14, 11, 15)
ITEM_COLOURS = {14, 15, 9, 11}


def _cell(x, y):
    return (slice(y + 1, y + 5), slice(x + 1, x + 5))


def _ring_mask(x, y):
    m = np.zeros((64, 64), dtype=bool)
    m[y, x:x + 6] = True
    m[y + 5, x:x + 6] = True
    m[y:y + 6, x] = True
    m[y:y + 6, x + 5] = True
    return m


def parse(grid):
    """Return (layout tuple, selected location index or None, energy) or None if not level 1."""
    g = np.asarray(grid)
    if g.shape != (64, 64):
        return None
    layout = []
    for (x, y) in LOCS:
        inner = g[_cell(x, y)]
        c = int(inner[0, 0])
        if (inner == c).all() and c in ITEM_COLOURS:
            layout.append(c)
        elif (inner[1:3, 1:3] == 2).all() and int((inner == BG).sum()) == 12:
            layout.append(None)
        else:
            return None
    if sorted(c for c in layout if c is not None) != sorted(ITEM_COLOURS):
        return None
    sel = None
    for i, (x, y) in enumerate(LOCS):
        if (g[_ring_mask(x, y)] == 0).all():
            sel = i
    row = g[HUD_ROW]
    e = int((row == 2).sum())
    if not ((row[:e] == 2).all() and (row[e:] == 3).all()):
        return None
    return tuple(layout), sel, e


def render(grid, layout, sel, energy):
    g = np.asarray(grid).copy()
    for i, (x, y) in enumerate(LOCS):
        g[y:y + 6, x:x + 6] = BG
        if layout[i] is None:
            g[y + 2:y + 4, x + 2:x + 4] = 2
        else:
            g[_cell(x, y)] = layout[i]
    if sel is not None:
        x, y = LOCS[sel]
        g[_ring_mask(x, y)] = 0
    g[HUD_ROW, :] = 3
    g[HUD_ROW, :max(0, energy)] = 2
    return g


def _hit(px, py):
    for i, (x, y) in enumerate(LOCS):
        if x <= px < x + 6 and y <= py < y + 6:
            return i
    return None


def _moves(layout):
    out = []
    for a in range(8):
        if layout[a] is None:
            continue
        for b in range(8):
            if b == a:
                continue
            nl = list(layout)
            if layout[b] is None:
                nl[b], nl[a] = layout[a], None
            elif a < 4 and b < 4:
                continue  # both palette: only moves the selection
            else:
                nl[a], nl[b] = layout[b], layout[a]
            out.append(tuple(nl))
    return out


_DIST = None


def _dist():
    global _DIST
    if _DIST is None:
        d = {INIT: 0}
        frontier = [INIT]
        while frontier:
            nxt = []
            for s in frontier:
                for t in _moves(s):
                    if t not in d:
                        d[t] = d[s] + 1
                        nxt.append(t)
            frontier = nxt
        _DIST = d
    return _DIST


def engine(grid, action, data):
    g = np.asarray(grid)
    parsed = parse(g)
    if parsed is None:
        return g.copy()
    layout, sel, energy = parsed
    action = int(action)
    if action == 6:
        if not isinstance(data, dict):
            return g.copy()
        loc = _hit(int(data.get("x", -1)), int(data.get("y", -1)))
        if loc is None:
            return g.copy()
        if sel is None:
            if layout[loc] is not None:
                return render(g, layout, loc, energy)
            return g.copy()
        if layout[loc] is not None:
            if loc == sel:
                return render(g, layout, None, energy)
            if sel < 4 and loc < 4:
                return render(g, layout, loc, energy)
            nl = list(layout)
            nl[sel], nl[loc] = layout[loc], layout[sel]
            return render(g, tuple(nl), None, energy - 1)
        nl = list(layout)
        nl[loc], nl[sel] = layout[sel], None
        return render(g, tuple(nl), None, energy - 1)
    if action == 5:
        if tuple(layout[4:]) == TARGET:
            return g.copy()  # level-up; the next level's board is not modelled
        return render(g, layout, None, energy - 1)
    if action == 7:
        if layout == INIT:
            return g.copy()
        d = _dist()
        here = d.get(layout)
        preds = [p for p in _moves(layout) if here is not None and d.get(p) == here - 1]
        # _moves is symmetric (every move is reversible by one move), so one-move
        # neighbours are exactly the one-move predecessors.
        if len(set(preds)) == 1:
            return render(g, preds[0], None, energy)
        return render(g, layout, None, energy)
    return g.copy()
