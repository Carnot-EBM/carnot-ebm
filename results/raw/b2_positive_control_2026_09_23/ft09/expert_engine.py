"""Expert grid-only engine for ft09 level 0 (level name "THR"), written from the public game source
environment_files/ft09/0d8bbf25/ft09.py. Uses ONLY (grid, action, data) plus the constants below.
No file reads, no simulator, no recorded grids, no lookup tables.

Constants and where they come from (ft09.py):
  - level 0 data {"kCv": 32, "cwU": [9, 8], "elp": center-only}          (levels[0], ~line 2064)
  - Hkx clickable tiles (3x3 grid cells) at the 8 positions in levels[0]    (~lines 2043-2050)
  - camera: 32x32 level rendered at scale 2, no letterbox (Camera.display_to_grid)
  - HUD (class sve.render_interface): row 63, first round(64*dzy/kCv) cells = 12, rest = 11;
    one decrement of dzy per non-winning tile click (Ft09.step -> self.lpw.lph()).
  - clicks that hit no tile leave the final frame unchanged (the UEq flash animation ends on its
    original color 2), and they do not decrement the HUD.
  - win check cgj() against the bsT sprite wmW at (22,22), pattern [[0,2,2],[0,8,0],[0,2,2]].
"""
import numpy as np

SCALE = 2  # display pixels per level cell (64 / 32)
TILES = [(18, 18), (22, 18), (26, 18), (18, 22), (26, 22), (18, 26), (22, 26), (26, 26)]  # (x, y)
CYCLE = [9, 8]  # cwU: a click advances the tile center color to the next entry
KCV = 32  # HUD capacity (clicks before GAME_OVER)
HUD_ROW = 63
HUD_ON, HUD_OFF = 12, 11
WIN_SPRITE = (22, 22)
WIN_PATTERN = [[0, 2, 2], [0, 8, 0], [0, 2, 2]]  # 0 -> neighbor center == 8, else != 8


def _tile_center_color(g, tx, ty):
    return int(g[SCALE * (ty + 1), SCALE * (tx + 1)])


def _hud_count(g):
    row = g[HUD_ROW]
    n = 0
    while n < row.shape[0] and int(row[n]) == HUD_ON:
        n += 1
    return n


def _dzy_from_hud(g):
    # GoJ = round(64 * dzy / KCV); with KCV=32 this is exactly 2*dzy, so dzy is recoverable.
    return int(round(_hud_count(g) * KCV / 64.0))


def _render_hud(g, dzy):
    goj = int(round(64 * dzy / KCV))
    g[HUD_ROW, :] = HUD_OFF
    g[HUD_ROW, :goj] = HUD_ON


def _is_win(g):
    wx, wy = WIN_SPRITE
    need = WIN_PATTERN[1][1]
    for j in range(3):
        for i in range(3):
            if i == 1 and j == 1:
                continue
            nx, ny = wx + (i - 1) * 4, wy + (j - 1) * 4
            if (nx, ny) not in TILES:
                continue
            c = _tile_center_color(g, nx, ny)
            ok = (c == need) if WIN_PATTERN[j][i] == 0 else (c != need)
            if not ok:
                return False
    return True


def engine(grid, action, data):
    g = np.array(grid, copy=True)
    if int(action) != 6:
        return g
    data = data or {}
    x, y = int(data.get("x", 0)), int(data.get("y", 0))
    gx, gy = x // SCALE, y // SCALE
    on_screen = 0 <= x < 64 and 0 <= y < 64
    hit = None
    if on_screen:
        for tx, ty in TILES:
            if tx <= gx < tx + 3 and ty <= gy < ty + 3:
                hit = (tx, ty)
                break
        if hit is None:
            return g  # no tile: no net change (flash animation restores color 2), no HUD tick
    if hit is not None:
        tx, ty = hit
        old = _tile_center_color(g, tx, ty)
        if old in CYCLE:
            new = CYCLE[(CYCLE.index(old) + 1) % len(CYCLE)]
            block = g[SCALE * ty: SCALE * (ty + 3), SCALE * tx: SCALE * (tx + 3)]
            block[block == old] = new
    if _is_win(g):
        return g  # level-up: next layout is not predictable from this level; row is not graded
    dzy = _dzy_from_hud(g)
    _render_hud(g, max(0, dzy - 1))
    return g


def is_level_complete(grid):
    return _is_win(np.asarray(grid))
