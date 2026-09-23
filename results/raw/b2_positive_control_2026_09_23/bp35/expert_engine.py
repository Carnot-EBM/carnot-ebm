"""Expert grid-only engine for bp35 level 1 (engine(grid, action, data) -> next_grid).

Uses ONLY its three arguments plus constants copied from the public game source
(environment_files/bp35/0a0ad940/bp35.py): sprite pixel art, the colour enum, the level-1
map `tjdtolkmxo["grid1"]`, and the rules in uakietkqfso / Bp35.urzvqcxbsz.
It reads no files, calls no simulator, and holds no recorded grids.

Known blind spots, by construction (hidden state the grid does not show):
  * ACTION7 is undo. The undo stack is not on screen, so we return the board unchanged
    and only advance the step counter.
  * The crusher (aknlbboysnc / jcyhkseuorf) is only modelled when it is visible. Its
    advance parity is taken from the step counter, which equals the hidden move count
    only when no undo has popped a move.
"""
import numpy as np

# ---- colours (bp35.py lines 3216-3233: the 16 names are range(16)) ----
BG = 10          # jltzfsatusf, BACKGROUND_COLOR
C5 = 5           # nplbvxmrmhi
C3 = 3           # mxsayyrckip
HUD_OFF = 0      # jptccilwmwb
HUD_ON = 15      # ebbemwaevan


def _sprite(rows, cmap, flip=False):
    rows = rows[::-1] if flip else rows
    w = max(len(r) for r in rows)
    a = np.full((len(rows), w), -1, dtype=np.int16)
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            if ch in cmap:
                a[y, x] = cmap[ch]
    return a


# bp35.py lines 3235-3383 (ymmwcccrhb)
WALL = _sprite(["xxxxxxx", "xxoxxxx", "xxxxxxx", "xxxxxxx", "xoxxxxx", "xxxxoxx", "xxxxxxx"],
               {"x": 5, "o": 3})
EBLK = _sprite([".......", ".oxxxo.", ".xxxxx.", ".xxxxx.", ".xxxxx.", ".oxxxo.", "......."],
               {"x": 14, "o": 3, ".": 5})
PL_R = _sprite(["  ... ", " ..x..", " .xxr.", " .xxr.", " ..x..", "  ... ", "       "],
               {"x": 9, ".": 5, "r": 11})
PL_L = _sprite(["  ... ", " ..x..", " .rxx.", " .rxx.", " ..x..", "  ... ", "       "],
               {"x": 9, ".": 5, "r": 11})
GEM = _sprite(["  ... ", " ..x..", " .xxx.", " ..x..", "  ... ", "      ", "      "],
              {"x": 7, ".": 5})
CR_M = _sprite([".ooooo", "..oo.o", "......", " .... ", "      ", "      ", "      "],
               {"o": 15, ".": 5}, flip=True)
CR_W = _sprite(["oooo.o", "o.oooo", "o.o.oo", "o.o.oo", "o.oooo", "oooo.o", "oooo.o"],
               {"o": 15, ".": 5}, flip=True)

# Level-1 map, bp35.py tjdtolkmxo["grid1"]; listing is reversed, so gy = 35 - listing_row.
_LISTING = [
    "wwwwwwwwwww", "wwwwwwwwwww", "wwwwwwwwwww", "wwwwwwwwwww", "wwwwwwwwwww",
    "mmmmmmmmmmm", "oo       oo", "oo       oo", "oo       oo", "oo       oo",
    "oo       oo", "oo       oo", "oo n     oo", "ooooooo ooo", "oo       oo",
    "oo       oo", "oooooxxxxoo", "ooooo    oo", "ooxxx    oo", "ooxxx    oo",
    "ooxxxoooooo", "oo       oo", "oo       oo", "oo  xxx  oo", "oo       oo",
    "oo       oo", "oooooxxxooo", "oo       oo", "oo +     oo", "ooooooooooo",
    "ooooooooooo", "ooooooooooo", "ooooooooooo", "ooooooooooo", "ooooooooooo",
    "ooooooooooo",
]
MAP = _LISTING[::-1]          # MAP[gy][gx]
H, W = len(MAP), len(MAP[0])  # 36 x 11
CELL = 6
NR, NC = 11, 11               # screen lattice rows R=0..10 (row 60-62 partial), cols C=0..10


def _map_cell(gx, gy):
    if gx < 0 or gx >= W or gy < 0 or gy >= H:
        return "o" if gx < 0 or gx >= W else " "
    ch = MAP[gy][gx]
    return ch if ch in "ox+" else " "   # player/crusher are objects, not tiles


def _blit(canvas, spr, top, left):
    h, w = spr.shape
    for yy in range(h):
        y = top + yy
        if y < 0 or y > 62:          # row 63 is the HUD
            continue
        for xx in range(w):
            x = left + xx
            if 0 <= x < 64 and spr[yy, xx] >= 0:
                canvas[y, x] = spr[yy, xx]


def _match(grid, spr, top, left, rows=None):
    """True if every opaque sprite pixel inside the view (rows 0..62) equals the grid."""
    h, w = spr.shape
    seen = 0
    for yy in range(h) if rows is None else rows:
        y = top + yy
        if y < 0 or y > 62:
            continue
        for xx in range(w):
            x = left + xx
            if 0 <= x < 64 and spr[yy, xx] >= 0:
                seen += 1
                if grid[y, x] != spr[yy, xx]:
                    return False
    return seen > 0


def _parse(g):
    """Screen lattice -> scene. Returns None when the frame does not parse."""
    # player: colour 9 at sprite (1,3)
    player = None
    for y, x in zip(*np.nonzero(g[:63] == 9)):
        top, left = y - 1, x - 3
        if top % CELL or left % CELL:
            continue
        for spr, face in ((PL_R, True), (PL_L, False)):
            # rows 0-2 carry the facing pixel; lower rows may be covered by the crusher (layer 12)
            if _match(g, spr, top, left) or _match(g, spr, top, left, rows=range(0, 3)):
                player = (left // CELL, top // CELL, face)
                break
        if player:
            break
    if player is None:
        return None
    tiles = {}
    for R in range(NR):
        for C in range(NC):
            top, left = R * CELL, C * CELL
            inner = g[top + 1:min(top + 6, 63), left + 1:min(left + 6, 64)]
            if inner.size == 0:
                tiles[(C, R)] = " "
                continue
            if _match(g, WALL, top, left, rows=range(1, 6)):
                tiles[(C, R)] = "o"
            elif _match(g, EBLK, top, left, rows=range(1, 6)):
                tiles[(C, R)] = "x"
            else:
                tiles[(C, R)] = " "
    gem = None
    for y, x in zip(*np.nonzero(g[:63] == 7)):
        top, left = y - 1, x - 3
        if top % CELL == 0 and left % CELL == 0 and _match(g, GEM, top, left):
            gem = (left // CELL, top // CELL)
            break
    crusher_R = None
    ys = np.nonzero((g[:63] == 15).any(axis=1))[0]
    if ys.size:
        # m sprite: its first opaque row is sprite row 3 (" .... "), colour 5 only; the first
        # colour-15 row of m is sprite row 5. Search R so that the m sprite fits.
        for R in range(-1, NR + 1):
            # sprite row 6 of m is painted over by the w block below it (same layer, drawn later)
            if _match(g, CR_M, R * CELL, 0, rows=range(3, 6)):
                crusher_R = R
                break
        if crusher_R is None:
            return None
    # world offset k (screen R=0 <-> gy k): walls never change, so match them to the map.
    ks = []
    for k in range(-2, H):
        ok = True
        for (C, R), t in tiles.items():
            if crusher_R is not None and R >= crusher_R:
                continue
            m = _map_cell(C, k + R)
            if (t == "o") != (m == "o"):
                ok = False
                break
            if t == "x" and m != "x":
                ok = False
                break
        if ok and _map_cell(player[0], k + player[1]) in (" ", "+"):
            ks.append(k)
    if len(ks) != 1:
        return None
    if crusher_R is not None:
        # Cells under the crusher are hidden by it; they were never clickable, so use the map.
        for R in range(max(crusher_R, 0), NR):
            for C in range(NC):
                m = _map_cell(C, ks[0] + R)
                tiles[(C, R)] = m if m in "ox" else " "
    return {"player": player, "tiles": tiles, "gem": gem, "k": ks[0], "crusher_R": crusher_R}


def _occ(scene, C, gy):
    """Occupancy of a world cell: parsed screen content when visible, else the original map."""
    k = scene["k"]
    R = gy - k
    if (C, R) in scene["tiles"] and 0 <= R < NR:
        t = scene["tiles"][(C, R)]
        if t == " " and scene["gem"] == (C, R):
            return "+"
        return t
    return _map_cell(C, gy)


def _render(scene, counter):
    g = np.full((64, 64), BG, dtype=np.int16)
    k = scene["k"]
    # gem (layer 9) first
    for R in range(-1, NR + 1):
        for C in range(NC):
            if _occ(scene, C, k + R) == "+":
                _blit(g, GEM, R * CELL, C * CELL)
    for R in range(-1, NR + 1):          # tiles (layer 10); shared border colour is 5 for both
        for C in range(NC):
            t = _occ(scene, C, k + R)
            if t == "o":
                _blit(g, WALL, R * CELL, C * CELL)
            elif t == "x":
                _blit(g, EBLK, R * CELL, C * CELL)
    pc, pr, face = scene["player"]
    _blit(g, PL_R if face else PL_L, pr * CELL, pc * CELL)   # layer 11
    cr = scene["crusher_R"]
    if cr is not None:                                          # layer 12: m, then w on top
        for C in range(NC):
            _blit(g, CR_M, cr * CELL, C * CELL)
        for C in range(NC):
            for d in range(1, 6):
                _blit(g, CR_W, (cr + d) * CELL, C * CELL)
    g[63, :] = HUD_OFF
    g[63, :min(counter, 64)] = HUD_ON
    return g


def _counter(g):
    row = g[63]
    n = 0
    while n < 64 and row[n] == HUD_ON:
        n += 1
    return n


def _fall(scene, C, gy_start):
    """fsvnqdbzrp with gravity toward smaller gy: first blocked cell above gy_start."""
    last, gy, n = gy_start, gy_start - 1, 0
    while _occ(scene, C, gy) == " " and gy >= 0:
        last, gy, n = gy, gy - 1, n + 1
    return n, last, _occ(scene, C, gy) == "+"


def _recentre(scene, new_player_gy):
    """Camera moves to player_gy*6-36, so screen R=0 becomes gy (player_gy-6)."""
    k_old = scene["k"]
    known = {}
    for (C, R), t in scene["tiles"].items():
        known[(C, k_old + R)] = t
    k_new = new_player_gy - 6
    tiles = {}
    for R in range(NR):
        for C in range(NC):
            gy = k_new + R
            tiles[(C, R)] = known.get((C, gy), _map_cell(C, gy) if _map_cell(C, gy) != "+" else " ")
    gem = None
    for R in range(NR):
        for C in range(NC):
            if _map_cell(C, k_new + R) == "+":
                gem = (C, R)
    cr = scene["crusher_R"]
    return {"player": scene["player"], "tiles": tiles, "gem": gem, "k": k_new,
            "crusher_R": None if cr is None else cr + (k_old - k_new)}


def engine(grid, action, data=None):
    g = np.asarray(grid).astype(np.int16)
    out_counter = _counter(g) + 1
    scene = _parse(g)
    if scene is None:
        out = g.copy()
        out[63, :] = HUD_OFF
        out[63, :min(out_counter, 64)] = HUD_ON
        return out
    a = int(action)
    pc, pr, face = scene["player"]
    k = scene["k"]
    if a in (3, 4):
        dx = 1 if a == 4 else -1
        scene["player"] = (pc, pr, dx > 0)
        tx = pc + dx
        tgt = "o" if tx < 0 else _occ(scene, tx, k + pr)
        crusher_step = (out_counter % 2 == 0)   # hidden move count assumed == step counter
        if tgt == " ":
            n, last_gy, gem_hit = _fall(scene, tx, k + pr)
            if n == 0:
                scene["player"] = (tx, pr, dx > 0)
                if crusher_step and scene["crusher_R"] is not None:
                    scene["crusher_R"] -= 1
            else:
                scene["player"] = (tx, pr, dx > 0)
                scene = _recentre(scene, last_gy)
                scene["player"] = (tx, last_gy - scene["k"], dx > 0)
        elif tgt == "+":
            scene["player"] = (tx, pr, dx > 0)
        else:  # blocked: bump animation nets to zero, facing updated, crusher may step
            if crusher_step and scene["crusher_R"] is not None:
                scene["crusher_R"] -= 1
        return _render(scene, out_counter)
    if a == 6 and isinstance(data, dict):
        x, y = int(data.get("x", -1)), int(data.get("y", -1))
        C, R = x // CELL, y // CELL          # camera is a multiple of 6 on level 1
        if (C, R) in scene["tiles"] and scene["tiles"][(C, R)] == "x":
            above = (C == pc and R == pr - 1)
            scene["tiles"][(C, R)] = " "
            if above:
                n, last_gy, gem_hit = _fall(scene, pc, k + pr - 1)
                scene = _recentre(scene, last_gy)
                scene["player"] = (pc, last_gy - scene["k"], face)
        return _render(scene, out_counter)
    # ACTION7 (undo) and anything else: board unchanged (history is not on screen).
    return _render(scene, out_counter)


def is_level_complete(grid):
    return False
