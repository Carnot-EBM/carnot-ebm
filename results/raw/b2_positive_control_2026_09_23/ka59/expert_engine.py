"""Hand-written grid-only world model for ka59 LEVEL 0 (positive control, B2 v3).

Contract: engine(grid, action, data) -> next_grid. Uses ONLY its three arguments plus
constants read off the public game source (environment_files/ka59/38d34dbb/ka59.py):
level 0 layout, colors, step size 3, StepCounter 100, camera padding 9. It reads no
files, calls no simulator, and embeds no recorded grid.

The one thing it cannot see is the hidden step counter. The HUD bar on row 63 shows
round(64 * c / 100); two counter values share most bar lengths. HUD_CONVENTION picks
which one to assume. It was fixed on the visible rows before any held-out score.
"""
import numpy as np

# ---- constants from the game source -------------------------------------------------
BG, BORDER, TARGET, RING, WALL = 1, 2, 4, 14, 15      # BACKGROUND_COLOR, PADDING/border, target, izokdsdgdo, wall
SEL_C, DESEL_C, FRESH_C = 0, 4, 5                        # gkaipxrkmo, armiwmeidu, sprite default centre
HUD_ON, HUD_OFF, HUD_ROW = 4, 0, 63                      # ckawyvsxuv.render_interface
STEP = 3                                                 # zsqdfmgyjo
K = 100                                                  # level 0 data StepCounter
PAD = 9                                                  # int((64 - 45*1) / 2), 45x45 level at scale 1
GRID_W = GRID_H = 45
# Level 0 sprites, (x, y) in level coords -> frame (row = y + PAD, col = x + PAD).
TARGETS = [(23 + PAD, 2 + PAD), (17 + PAD, 35 + PAD)]    # 0009ouocpihipp, 5x5 ring of 4, centre transparent
WALL_R0, WALL_C0, WALL_H, WALL_W = 12 + PAD, 24 + PAD, 21, 6   # 0014ysspdlqsqg
HUD_CONVENTION = "upper"   # assume the LARGEST counter value consistent with the bar

_STATIC = np.full((64, 64), BG, dtype=int)
for (tr, tc) in TARGETS:
    _STATIC[tr:tr + 5, tc:tc + 5] = TARGET
    _STATIC[tr + 1:tr + 4, tc + 1:tc + 4] = BG
_STATIC[WALL_R0:WALL_R0 + WALL_H, WALL_C0:WALL_C0 + WALL_W] = WALL
_WALL = np.zeros((64, 64), dtype=bool)
_WALL[WALL_R0:WALL_R0 + WALL_H, WALL_C0:WALL_C0 + WALL_W] = True

_SIDES = {"top": [(0, 0), (0, 1), (0, 2)], "bottom": [(2, 0), (2, 1), (2, 2)],
          "left": [(0, 0), (1, 0), (2, 0)], "right": [(0, 2), (1, 2), (2, 2)]}
_RING = [(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]


def _ring_matches(win, zeroed):
    zero_cells = set()
    for s in zeroed:
        zero_cells.update(_SIDES[s])
    return all(win[i, j] == (SEL_C if (i, j) in zero_cells else RING) for (i, j) in _RING)


def _parse_blocks(g):
    """Find 3x3 player blocks on the 3-cell lattice (rows/cols = 0 mod 3)."""
    blocks = []
    import itertools
    subsets = [z for n in range(0, 5) for z in itertools.combinations(_SIDES, n)]
    for r in range(0, 61, 3):
        for c in range(0, 62, 3):
            win = g[r:r + 3, c:c + 3]
            if win.shape != (3, 3):
                continue
            centre = int(win[1, 1])
            if centre in (DESEL_C, FRESH_C):
                if _ring_matches(win, ()):
                    blocks.append({"r": r, "c": c, "centre": centre, "sel": False})
            elif centre == SEL_C:
                if any(_ring_matches(win, z) for z in subsets if len(z) < 4):
                    blocks.append({"r": r, "c": c, "centre": centre, "sel": True})
    return blocks


def _overlap(a_r, a_c, b_r, b_c):
    return abs(a_r - b_r) < 3 and abs(a_c - b_c) < 3


def _hits(mask_or_grid, r, c, border_grid=None):
    """True if the 3x3 footprint at (r, c) touches a border cell (colour 2) or leaves the frame."""
    if r < 0 or c < 0 or r + 3 > 63 or c + 3 > 64:
        return True
    return bool((border_grid[r:r + 3, c:c + 3] == BORDER).any())


def _on_wall(r, c):
    if r < 0 or c < 0:
        return False
    return bool(_WALL[max(r, 0):r + 3, max(c, 0):c + 3].any())


def _push(blocks, idx, dr, dc, border_grid):
    """Pushed-block slide: up to 5 moves of 3, blocked by the border only (ifoelczjjh),
    and it keeps sliding past 5 moves while it sits on a wall (dgjbrykwhi)."""
    k = 0
    guard = 0
    while guard < 200:
        guard += 1
        b = blocks[idx]
        if k >= 5 and not _on_wall(b["r"], b["c"]):
            break
        nr, nc = b["r"] + dr, b["c"] + dc
        if _hits(None, nr, nc, border_grid):
            break
        blocked = False
        for j, o in enumerate(blocks):
            if j != idx and _overlap(nr, nc, o["r"], o["c"]):
                # chain push of one step; with two blocks this only hits the pusher, which never happens
                onr, onc = o["r"] + dr, o["c"] + dc
                if _hits(None, onr, onc, border_grid):
                    blocked = True
                else:
                    o["r"], o["c"] = onr, onc
        if blocked:
            break
        b["r"], b["c"] = nr, nc
        k += 1


def _hud_next(row):
    L = 0
    for v in row:
        if v == HUD_ON:
            L += 1
        else:
            break
    cands = [c for c in range(0, K + 1) if round(64 * (c / K)) == L]
    if not cands:
        return row.copy()
    c = max(cands) if HUD_CONVENTION == "upper" else min(cands)
    c2 = max(0, c - 1)
    L2 = round(64 * (c2 / K))
    out = np.full(64, HUD_OFF, dtype=int)
    out[:L2] = HUD_ON
    return out


def _render(grid, blocks):
    out = np.array(grid, dtype=int, copy=True)
    # erase every block to the static layer, then redraw
    for b in blocks:
        out[b["r0"]:b["r0"] + 3, b["c0"]:b["c0"] + 3] = _STATIC[b["r0"]:b["r0"] + 3, b["c0"]:b["c0"] + 3]
    sel = next((b for b in blocks if b["sel"]), None)
    for b in blocks:
        r, c = b["r"], b["c"]
        out[r:r + 3, c:c + 3] = RING
        out[r + 1, c + 1] = b["centre"]
    if sel is not None:
        r, c = sel["r"], sel["c"]
        for o in blocks:
            if o is sel:
                continue
            if o["c"] == c and o["r"] == r - 3:
                out[r, c:c + 3] = SEL_C
            if o["c"] == c and o["r"] == r + 3:
                out[r + 2, c:c + 3] = SEL_C
            if o["r"] == r and o["c"] == c - 3:
                out[r:r + 3, c] = SEL_C
            if o["r"] == r and o["c"] == c + 3:
                out[r:r + 3, c + 2] = SEL_C
    return out


def engine(grid, action, data=None):
    g = np.asarray(grid, dtype=int)
    blocks = _parse_blocks(g)
    for b in blocks:
        b["r0"], b["c0"] = b["r"], b["c"]
    action = int(action)
    sel_i = next((i for i, b in enumerate(blocks) if b["sel"]), None)
    if action == 6:
        d = data or {}
        x, y = int(d.get("x", 0)), int(d.get("y", 0))
        gx, gy = x - PAD, y - PAD
        if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
            hit = next((i for i, b in enumerate(blocks)
                        if b["r"] <= y < b["r"] + 3 and b["c"] <= x < b["c"] + 3), None)
            if hit is not None:
                if sel_i is not None and sel_i != hit:
                    blocks[sel_i]["sel"] = False
                    blocks[sel_i]["centre"] = DESEL_C
                blocks[hit]["sel"] = True
                blocks[hit]["centre"] = SEL_C
    elif action in (1, 2, 3, 4) and sel_i is not None:
        dr, dc = {1: (-STEP, 0), 2: (STEP, 0), 3: (0, -STEP), 4: (0, STEP)}[action]
        s = blocks[sel_i]
        nr, nc = s["r"] + dr, s["c"] + dc
        blocked = _hits(None, nr, nc, g) or _on_wall(nr, nc)
        if not blocked:
            pushed = [j for j, o in enumerate(blocks) if j != sel_i and _overlap(nr, nc, o["r"], o["c"])]
            if pushed:
                for j in pushed:
                    _push(blocks, j, dr, dc, g)
            else:
                s["r"], s["c"] = nr, nc
    out = _render(g, blocks)
    out[HUD_ROW] = _hud_next(g[HUD_ROW])
    return out


def is_level_complete(grid):
    g = np.asarray(grid, dtype=int)
    blocks = _parse_blocks(g)
    return all(any(b["r"] == tr + 1 and b["c"] == tc + 1 for b in blocks) for (tr, tc) in TARGETS)
