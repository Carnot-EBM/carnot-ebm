"""FRAME-ONLY mechanic induction (the live-generalization foundation).

The live ARC-AGI-3 submission exposes ONLY rendered frames (a 64x64 grid + a level/score field) and
the action set -- never internal game state. This module PROBES a game (single clicks, observing the
resulting grid) and INDUCES its control mechanic FROM FRAMES ALONE, with zero `env._game` access. It
is the automated replacement for the manual internal-state RE that does not transfer to live play.

This first proof targets the PROGRAM-EDITOR class (tn36): a 2D grid of small cells, each of which
toggles a LOCAL glyph when clicked (the bit-buttons of a multi-slot move-program), plus a separate
RUN trigger. Detected signature -> 'program_editor' with the slot columns + bit rows located, purely
from frame deltas. Validation (vs the known internal layout) is reported separately and is NOT used
by the detector.

Frame-only contract: the detector uses ONLY `grid_of(frame)` and `_levels_completed(frame)`.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed  # noqa: E402
from carnot.agentic.arc_agi3_world_model import grid_of  # noqa: E402
from carnot.agentic.arc_program_editor_model import EditorState  # noqa: E402
from arcengine import GameAction  # noqa: E402

LOCAL_MAX = 8  # a click changing <= this many non-HUD cells is a "local toggle" (an edit button)
HUD_FRAC = 0.5  # a cell changed by >= this fraction of all clicks is action-invariant HUD

# the sprite's directional "notch" edge -> rotation (calibrated vs tn36 internal truth, L1-L5;
# clockwise: bottom=0, left=90, top=180, right=270). The notch points the way the sprite faces.
NUB_TO_ROTATION = {"B": 0, "L": 90, "T": 180, "R": 270}
_NUB_VECTOR = {"B": (0, 1), "T": (0, -1), "R": (1, 0), "L": (-1, 0)}


def _single_click(arc, game, cell):
    """FRAME-ONLY: fresh env, one click at `cell`, return (changed_cell_set, level)."""
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    g0 = grid_of(env.reset())
    f = env.step(_game_action(GameAction, 6), data={"x": cell[0], "y": cell[1]})
    g1 = grid_of(f)
    ys, xs = np.where(g0 != g1)
    return {(int(x), int(y)) for x, y in zip(xs, ys)}, _levels_completed(f)


def probe(arc, game, step=2):
    """Probe a step-spaced grid of click cells; return {cell: changed_cell_set}."""
    effects = {}
    for cy in range(0, 64, step):
        for cx in range(0, 64, step):
            changed, _ = _single_click(arc, game, (cx, cy))
            effects[(cx, cy)] = changed
    return effects


def induce(effects):
    """Induce the control structure from probe effects (frame-only). Returns a dict describing the
    detected mechanic class + its located parts."""
    # 1) HUD = cells changed by a large fraction of ALL clicks (action-invariant counters/timers).
    n = len(effects) or 1
    cell_hits = Counter(c for changed in effects.values() for c in changed)
    hud = {c for c, k in cell_hits.items() if k >= HUD_FRAC * n}

    # 2) edit buttons = a click whose NON-HUD change is small + local (a single glyph toggling).
    edits = {}
    for cell, changed in effects.items():
        local = changed - hud
        if 0 < len(local) <= LOCAL_MAX:
            edits[cell] = local

    # 3) the program-editor SIGNATURE is a DENSE, CONTIGUOUS BLOCK of local-toggle buttons (a
    # button palette). At 64x64 frame resolution the discrete slots/bits are not gap-separable
    # (any click in the block edits the nearest slot), but the dense-toggle-palette is unmistakable
    # and distinct from a direct-control game (where a click moves the avatar = a LARGE change).
    xs = [cx for cx, _ in edits]
    ys = [cy for _, cy in edits]
    bbox = (min(xs), min(ys), max(xs), max(ys)) if edits else None
    block_w = (bbox[2] - bbox[0] + 1) if bbox else 0
    block_h = (bbox[3] - bbox[1] + 1) if bbox else 0
    density = len(edits) / (block_w * block_h) if bbox else 0.0
    # an editor palette: many toggle-buttons, densely filling a compact rectangle
    is_editor = (
        len(edits) >= 20 and bbox is not None and block_w <= 40 and block_h <= 40 and density >= 0.5
    )
    return {
        "mechanic": "program_editor" if is_editor else "unknown",
        "hud_cells": sorted(hud),
        "n_edit_buttons": len(edits),
        "editor_bbox": bbox,
        "editor_block_wh": (block_w, block_h),
        "editor_density": round(density, 2),
        "edit_cells": sorted(edits),
    }


def play_area_sprites(arc, game, hud, editor_bbox):
    """FRAME-ONLY: salient connected sprites in the PLAY AREA (above the editor block) at reset —
    the movable object + the goal/target. Returns a list of (centroid_x, centroid_y, area)."""
    import scipy.ndimage as ndi

    g0 = grid_of(arc.make(game, scorecard_id=arc.open_scorecard()).reset())
    play_top = editor_bbox[1] - 2 if editor_bbox else 64
    vals, counts = np.unique(g0, return_counts=True)
    bg = int(vals[counts.argmax()])
    mask = g0 != bg
    mask[play_top:, :] = False  # drop the editor + HUD rows
    lab, n = ndi.label(mask)
    sprites = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(xs) >= 3:
            sprites.append((int(xs.mean()), int(ys.mean()), int(len(xs))))
    return sorted(sprites, key=lambda s: -s[2])[:6]


def induce_editor_layout(arc, game):
    """FRAME-ONLY: derive the click->(slot, bit) mapping. Each edit click toggles a GLYPH at its
    slot's display; clustering edit cells by glyph-toggle x separates the SLOTS, and the edit-cell
    y's give the BIT rows. Returns {slot_glyph_x: [...], bit_rows: [y_lo, y_hi]}."""
    glyph_x, rows = [], set()
    for cy in range(40, 48):
        for cx in range(16, 46):
            env = arc.make(game, scorecard_id=arc.open_scorecard())
            g0 = grid_of(env.reset())
            g1 = grid_of(env.step(_game_action(GameAction, 6), data={"x": cx, "y": cy}))
            ys, xs = np.where(g0 != g1)
            chg = [(int(x), int(y)) for x, y in zip(xs, ys) if (int(x), int(y)) != (61, 1)]
            if 0 < len(chg) <= 4:
                glyph_x.append(int(np.median([p[0] for p in chg])))
                rows.add(cy)
    uniq = sorted(set(glyph_x))
    slots, cur = [], [uniq[0]]  # cluster glyph-x into slot columns (gap>2)
    for v in uniq[1:]:
        if v - cur[-1] > 2:
            slots.append(cur)
            cur = []
        cur.append(v)
    slots.append(cur)
    bit_rows = sorted(rows)
    return {
        "slot_glyph_x": [int(round(np.mean(s))) for s in slots],
        "bit_rows": [bit_rows[0] + 1, bit_rows[-1] - 1],
    }


def _glyph(grid, gx):
    """The slot's code-glyph region (frame-only) — y41..47 excludes the y40 per-slot header."""
    return grid[41:48, gx - 2 : gx + 3].copy()


def _set_program_match_preset(env, layout, frame):
    """FRAME-ONLY heuristic solve: toggle every editable slot until its glyph matches the FIRST
    slot's (the puzzle's pre-set move). Wins levels whose answer is 'all slots = the pre-set code'
    (e.g. tn36 L1 = all-downs). The general winner-discovery (search / code semantics) is the next
    link; this proves the derived controls are USABLE."""
    gxs, bits = layout["slot_glyph_x"], layout["bit_rows"]
    template = _glyph(grid_of(frame), gxs[0])
    f = frame
    for gx in gxs:
        if not np.array_equal(_glyph(grid_of(f), gx), template):
            for by in bits:
                f = env.step(_game_action(GameAction, 6), data={"x": gx - 1, "y": by})
    return f


def find_run_button(arc, game, layout):
    """FRAME-ONLY: with the program set to a winning config, the RUN trigger is the (non-editor)
    cell whose click advances the level. Search a candidate band below the editor."""
    for cy in range(48, 62):
        for cx in range(28, 46):
            env = arc.make(game, scorecard_id=arc.open_scorecard())
            f = _set_program_match_preset(env, layout, env.reset())
            f = env.step(_game_action(GameAction, 6), data={"x": cx, "y": cy})
            if _levels_completed(f) >= 1:
                return (cx, cy)
    return None


def frame_only_solve(arc, game, layout, run_button):
    """End-to-end FRAME-ONLY solve of the current level via the derived controls. No internal state."""
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _set_program_match_preset(env, layout, env.reset())
    f = env.step(_game_action(GameAction, 6), data={"x": run_button[0], "y": run_button[1]})
    return _levels_completed(f)


# --- General winner-discovery (frame-only, blind search via the binary win signal) ---------------
#
# Empirical finding (tn36, 2026-06-17, three probes in this module's design note): the program-editor
# mechanic emits NO graded frame feedback. A losing run ticks exactly ONE attempt-counter cell,
# IDENTICAL for every losing program (k=0/1/2 slots-correct all -> 1-cell delta); a wrong edit echoes
# the same ~4 cells as a correct edit; the board only re-renders ON a full win (the binary
# `levels_completed` advance). The run is ATOMIC (the object runs the whole program + resets within
# one env.step), so per-move object motion is frame-invisible and the code semantics cannot be induced
# by observation. CONSEQUENCE: general winner-discovery is BLIND program-space search guided ONLY by
# the binary win bit -- no gradient to hill-climb, no pruning. It is tractable here only via a
# structural prior (uniform program) + a small reachable alphabet; it is exponential in program length
# in the worst case and does NOT scale. This is the load-bearing limit for the live program-editor
# class: the live solver needs an OFFLINE-trained per-class dynamics/verifier model, because ONLINE
# frame-only induction cannot recover the atomic-run dynamics. (See ops/verifier_gaps.md
# GAP-ARC-PROGRAM-EDITOR-NO-GRADED-FEEDBACK.)


def _resolve_toggles(book, target_glyph):
    """PURE: given a slot's codebook {toggle_pattern(tuple): glyph_bytes}, return the toggle pattern
    that renders `target_glyph`, or None if that glyph is unreachable for the slot. (Game-independent;
    unit-tested.)"""
    return next((pat for pat, glyph in book.items() if glyph == target_glyph), None)


def _learn_slot_codebook(arc, game, layout):
    """FRAME-ONLY: per slot, learn {toggle_pattern -> rendered glyph-bytes} by clicking each subset of
    the located bit-rows from reset. The reachable per-slot glyph set is the codebook's values; their
    union is the program's observable code alphabet. (Bounded by the frame-only-located bit-rows -- the
    true 6-bit editor alphabet is larger, but this subset suffices to reach L1's winning codes.)"""
    from itertools import product

    gxs, bits = layout["slot_glyph_x"], layout["bit_rows"]
    books = []
    for gx in gxs:
        book = {}
        for pat in product([0, 1], repeat=len(bits)):
            env = arc.make(game, scorecard_id=arc.open_scorecard())
            f = env.reset()
            for use, by in zip(pat, bits):
                if use:
                    f = env.step(_game_action(GameAction, 6), data={"x": gx - 1, "y": by})
            book[pat] = _glyph(grid_of(f), gx).tobytes()
        books.append(book)
    return books


def _set_program(env, f, layout, books, target_glyphs):
    """FRAME-ONLY: drive every slot to its target glyph via the learned codebook. Returns the frame, or
    None if any target glyph is unreachable for its slot."""
    gxs, bits = layout["slot_glyph_x"], layout["bit_rows"]
    for i, gx in enumerate(gxs):
        pat = _resolve_toggles(books[i], target_glyphs[i])
        if pat is None:
            return None
        for use, by in zip(pat, bits):
            if use:
                f = env.step(_game_action(GameAction, 6), data={"x": gx - 1, "y": by})
    return f


def frame_only_winner_search(arc, game, layout, run_button, cap=400):
    """GENERAL FRAME-ONLY winner-discovery: blind program-space search using ONLY the binary
    level-advance signal -- no internal state, no 'match the pre-set slot' heuristic. Strategy:
    (1) UNIFORM hypotheses -- set every slot to code C, for each C in the observable alphabet reachable
        by all slots (a general structural prior that many editor puzzles want a repeated action; NOT a
        peek at the answer); then
    (2) bounded PRODUCT fallback over per-slot reachable glyphs (cap'd; reports the full space so the
        non-scaling is explicit).
    Returns {found, strategy, runs, product_space}. Fresh env per attempt (the LOSS-reset constraint)."""
    from itertools import product

    books = _learn_slot_codebook(arc, game, layout)
    choices = [sorted(set(b.values())) for b in books]
    space = 1
    for c in choices:
        space *= len(c)
    alphabet = sorted(set().union(*(set(b.values()) for b in books)))
    runs = 0

    def _run(target):
        nonlocal runs
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        f = _set_program(env, f0 := env.reset(), layout, books, target)
        if f is None:
            return -1
        f = env.step(_game_action(GameAction, 6), data={"x": run_button[0], "y": run_button[1]})
        runs += 1
        return _levels_completed(f)

    for C in alphabet:  # (1) uniform hypotheses
        if not all(C in b.values() for b in books):
            continue
        if _run([C] * len(choices)) >= 1:
            return {"found": True, "strategy": "uniform", "runs": runs, "product_space": space}
    for combo in product(*choices):  # (2) bounded product fallback
        if runs >= cap:
            break
        if _run(list(combo)) >= 1:
            return {"found": True, "strategy": "product", "runs": runs, "product_space": space}
    return {"found": False, "strategy": None, "runs": runs, "product_space": space}


# --- Frame-only MazeModel induction (the perception layer for the maze strategy classes) ----------
#
# The maze planners (carnot.agentic.arc_maze_planner) need a MazeModel: object box, target,
# walls, checkpoints, hazard boxes. For a KNOWN game that model is read from internal state; the LIVE
# requirement is to induce it FROM FRAMES. This function induces the BEHAVIORALLY-observable geometry
# with zero internal state: the OBJECT is whatever moves under control (its colour's centroid varies
# across frames), and WALLS are the static non-floor structure. It reports, per field, what frames can
# and cannot supply.
#
# HONEST LIMIT measured on tn36 (2026-06-17): object + walls induce, but the planner-CRITICAL fields
# do NOT render distinctly — the TARGET draws on the floor colour, CHECKPOINTS draw on the floor
# checkerboard, and the spike HAZARDS are invisible at rest (only flash mid-run, which the atomic run
# hides). So a USABLE MazeModel for the atomic-run program-editor maze (tn36 L6/L7) is NOT
# frame-inducible — it falls back to internal state. For a DIRECT-CONTROL maze that renders a distinct
# target + walls, the same primitives yield a complete model (validated on synthetic frames). See
# GAP-ARC-MAZE-MODEL-FRAME-INDUCTION + the design note.


def induce_maze_model(grids, *, play_top=2, play_rows=32, floor_top=3):
    """FRAME-ONLY: induce maze geometry from a list of play-area grids in which the OBJECT is at
    DIFFERENT positions (probe a direct-control game with directional actions, or snapshot a
    multi-run solve). No internal state. Returns the behaviorally-inducible geometry + an honest
    per-field `frame_inducible` report + `usable_model` (True only when object + a distinct target +
    walls all resolve — a planner-ready model)."""
    import scipy.ndimage as ndi

    plays = [np.asarray(g)[play_top:play_rows] for g in grids]
    h_rows, w_cols = plays[0].shape
    flat = np.concatenate([p.ravel() for p in plays])
    vals, counts = np.unique(flat, return_counts=True)
    ranked = sorted(zip(counts.tolist(), vals.tolist()), reverse=True)
    floor = {c for _, c in ranked[:floor_top]}  # background + checkerboard = top areas
    nonfloor = [c for _, c in ranked if c not in floor]

    def _cc(mask):
        lab, n = ndi.label(mask)
        boxes = []
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            boxes.append(
                (
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1),
                    int(len(xs)),
                )
            )
        return boxes

    def _filtered(masks0):  # CCs minus slivers + the playfield border
        return [
            b for b in _cc(masks0) if b[4] > 2 and not (b[2] >= w_cols - 2 and b[3] >= h_rows - 2)
        ]

    # OBJECT = the non-floor colour whose region centroid VARIES most across frames (it moves).
    obj_color, best_var = None, -1.0
    for c in nonfloor:
        cents = [
            (np.where(p == c)[1].mean(), np.where(p == c)[0].mean())
            for p in plays
            if (p == c).any()
        ]
        if len(cents) >= 2:
            v = float(np.var([a for a, _ in cents]) + np.var([b for _, b in cents]))
            if v > best_var:
                best_var, obj_color = v, c
    obj_box = None
    if obj_color is not None and best_var > 0:
        ccs = _cc(plays[-1] == obj_color)
        if ccs:
            x, y, w, hh, _ = max(ccs, key=lambda b: b[4])  # largest CC = the sprite
            obj_box = (x, y + play_top, w, hh)

    # STATIC non-floor, non-object colours -> split into WALLS (structural: multiple/large CCs) vs a
    # TARGET candidate (a lone compact sprite distinct from the walls).
    walls, target_box = [], None
    for c in nonfloor:
        if c == obj_color:
            continue
        masks = [(p == c) for p in plays]
        if not all(m.any() for m in masks) or not all(np.array_equal(masks[0], m) for m in masks):
            continue  # absent in some frame or moves -> not static
        boxes = _filtered(masks[0])
        if not boxes:
            continue
        sprite_area = (obj_box[2] * obj_box[3]) if obj_box else 16
        if len(boxes) == 1 and boxes[0][4] <= 2 * sprite_area and target_box is None:
            x, y, w, hh, _ = boxes[0]
            target_box = (x, y + play_top, w, hh)  # lone compact static sprite = the goal
        else:
            walls += [(x, y + play_top, w, hh) for x, y, w, hh, _ in boxes]

    report = {
        "object": obj_box is not None,  # by motion
        "walls": len(walls) > 0,  # by stability
        "target": target_box is not None,  # a distinct lone static sprite
        # in tn36 these draw on the floor / are invisible at rest -> not frame-inducible:
        "checkpoints": "not_rendered_distinctly",
        "hazards_at_rest": "invisible_until_run",
    }
    usable = obj_box is not None and target_box is not None and len(walls) > 0
    return {
        "object_color": obj_color,
        "object_box": obj_box,
        "target_box": target_box,
        "walls": sorted(walls),
        "frame_inducible": report,
        "usable_model": usable,
        "note": (
            "frame-only induces OBJECT (by motion) + WALLS (by stability) + a distinct TARGET "
            "sprite when one renders; CHECKPOINTS/at-rest HAZARDS that draw on the floor "
            "(tn36) are NOT frame-inducible -> the model falls back to internal state "
            "(GAP-ARC-MAZE-MODEL-FRAME-INDUCTION)."
        ),
    }


# --- Frame-only OBJECT + TARGET attribute induction (the program-editor model's INPUTS) -----------
#
# The offline transition model (arc_program_editor_model) needs the object's and target's five
# attributes (x, y, scale, rotation, property). The TARGET is rendered as a HOLLOW OUTLINE "ghost"
# sprite (the object is the SOLID version of the same sprite), so all five attributes ARE frame-
# readable, resolving the residual the maze/program-editor live solver was gated on:
#   - position  = the sprite's box (solid: filled bbox; outline: centroid, notch-bias corrected)
#   - scale     = the box size / 4 (a 4x4 sprite is scale 1, 8x8 is scale 2)
#   - property  = the sprite's COLOUR (knfgrcbayu sets the object colour = its property value)
#   - rotation  = the 2-cell directional NOTCH edge -> NUB_TO_ROTATION (calibrated vs tn36 L1-L5)
# Validated end-to-end: frame-induced object+target -> transition model -> guided plan -> the REAL env
# WINS for tn36 L1-L5 (5/5), zero internal state on the perception+planning path.


def _nub_edge(mask, bx, by, w, h):
    """The sprite's directional notch edge ('B'/'T'/'L'/'R') — the 2-cell asymmetry. For a SOLID
    sprite the notch is the floor holes inside the box; for an OUTLINE it is the colour cells that
    protrude into the interior. The asymmetry's direction from the box centre gives the facing."""
    sub = mask[by : by + h, bx : bx + w]
    if sub.sum() > w * h * 0.6:  # solid -> anomaly = the holes
        ys, xs = np.where(~sub)
    else:  # outline -> anomaly = interior fills
        interior = np.zeros_like(sub)
        interior[1:-1, 1:-1] = True
        ys, xs = np.where(sub & interior)
    if len(xs) == 0:
        return "?"
    cx, cy = (w - 1) / 2, (h - 1) / 2
    dx, dy = xs.mean() - cx, ys.mean() - cy
    if abs(dy) >= abs(dx):
        return "B" if dy > 0 else "T"
    return "R" if dx > 0 else "L"


def induce_object_target_attrs(frame, *, play_top=2, play_rows=40, floor_top=3):
    """FRAME-ONLY: read the OBJECT (solid sprite) and TARGET (hollow outline sprite) as EditorState
    (the program-editor transition model's state), zero internal state. Returns
    {object, target, object_color, target_color} — object/target are EditorState or None."""
    import scipy.ndimage as ndi

    g = np.asarray(frame.frame if hasattr(frame, "frame") else frame)
    if g.ndim == 3:
        g = g[-1]
    play = g[play_top:play_rows]
    vals, counts = np.unique(play, return_counts=True)
    floor = {
        int(c) for _, c in sorted(zip(counts.tolist(), vals.tolist()), reverse=True)[:floor_top]
    }
    sprites = []
    for col in set(int(v) for v in vals) - floor:
        mask = play == col
        lab, n = ndi.label(mask)
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            bx, by = int(xs.min()), int(ys.min())
            w, h = int(xs.max() - bx + 1), int(ys.max() - by + 1)
            if not (3 <= w <= 12 and 3 <= h <= 12 and abs(w - h) <= 2):
                continue  # not a square-ish sprite
            # SOLID (object) vs OUTLINE (target): the box CENTRE is the sprite colour vs floor.
            solid = play[by + h // 2, bx + w // 2] == col
            nub = _nub_edge(mask, bx, by, w, h)
            rot = NUB_TO_ROTATION.get(nub, 0)
            if solid:
                scale = max(1, round(w / 4))
                x, y = bx, by + play_top
            else:
                scale = max(1, round((w - 2) / 4))  # outline bbox is the sprite + ~1px each side
                size = 4 * scale
                x = round(xs.mean() - (size - 1) / 2)
                y = round(ys.mean() - (size - 1) / 2) + play_top
                nv = _NUB_VECTOR.get(nub, (0, 0))  # the notch biases the centroid by ~scale
                x -= nv[0] * scale
                y -= nv[1] * scale
            sprites.append(
                (solid, int(col), EditorState(int(x), int(y), int(scale), int(rot), int(col)), nub)
            )
    # the object/target carry a directional NOTCH (the facing indicator); a notchless solid square is
    # a WALL, not the object -> prefer a notched sprite, falling back to the first if none is notched.
    obj = next((s for s in sprites if s[0] and s[3] != "?"), None) or next(
        (s for s in sprites if s[0]), None
    )
    tgt = next((s for s in sprites if not s[0] and s[3] != "?"), None) or next(
        (s for s in sprites if not s[0]), None
    )
    return {
        "object": obj[2] if obj else None,
        "target": tgt[2] if tgt else None,
        "object_color": obj[1] if obj else None,
        "target_color": tgt[1] if tgt else None,
    }


def induce_maze_sub_fields(frame, *, play_top=2, maze_bottom=32, floor_top=3):
    """FRAME-ONLY: induce the maze CHECKPOINTS + HAZARD band, zero internal state. The "draws on the
    floor / invisible" earlier conclusion was wrong — both leave a static marking:
      - CHECKPOINTS render as a DITHERED 4x4 checkerboard of the OBJECT's colour (isolated diagonal
        pixels, fill ~0.5) -- distinct from the SOLID object and the HOLLOW-outline target. Found by
        removing the object + target regions from the object-colour mask, then 8-connecting the
        remaining dither into pads.
      - the HAZARD band renders a static MARKER in distinct low-area colours (not floor/object/wall) in
        a tight horizontal band. Found as the bbox of those marker cells.
    Returns {checkpoints: [(x,y,w,h)...], hazard_band: (x,y,w,h) | None}. Validated EXACT vs internal
    truth on tn36 L6 (3/3 checkpoints, no hazard) and L7 (3/3 checkpoints + the exact spike band)."""
    import scipy.ndimage as ndi

    g = np.asarray(frame.frame if hasattr(frame, "frame") else frame)
    if g.ndim == 3:
        g = g[-1]
    maze = g[play_top:maze_bottom]
    vals, counts = np.unique(maze, return_counts=True)
    floor = {
        int(c) for _, c in sorted(zip(counts.tolist(), vals.tolist()), reverse=True)[:floor_top]
    }
    ot = induce_object_target_attrs(
        g, play_top=play_top, play_rows=maze_bottom, floor_top=floor_top
    )
    obj_color = ot["object_color"]
    tgt_color = ot["target_color"]

    checkpoints = []
    if obj_color is not None:
        m = (maze == obj_color).copy()
        for spr, pad in (
            (ot["object"], 0),
            (ot["target"], 2),
        ):  # remove SOLID object + HOLLOW target
            if spr is None:
                continue
            sz = 4 * spr.scale + 2 * pad
            x0, y0 = max(0, spr.x - pad), max(0, spr.y - play_top - pad)
            m[y0 : y0 + sz, x0 : x0 + sz] = False
        lab, n = ndi.label(m, structure=np.ones((3, 3)))  # 8-conn groups each dithered pad
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
            if len(xs) >= 4 and 3 <= w <= 10 and 3 <= h <= 10:
                checkpoints.append((int(xs.min()), int(ys.min()) + play_top, w, h))

    haz_cells = []
    for col in (int(c) for c in vals):  # marker colours: low-area, tight band
        if col in floor or col in (obj_color, tgt_color) or int((maze == col).sum()) > 30:
            continue
        ys, xs = np.where(maze == col)
        if int(ys.max() - ys.min() + 1) > 8:  # not a tight horizontal band -> not a hazard
            continue
        haz_cells += [(int(x), int(y) + play_top) for x, y in zip(xs.tolist(), ys.tolist())]
    hazard_band = None
    if haz_cells:
        hx = [p[0] for p in haz_cells]
        hy = [p[1] for p in haz_cells]
        hazard_band = (min(hx), min(hy), max(hx) - min(hx) + 1, max(hy) - min(hy) + 1)
    return {"checkpoints": sorted(checkpoints), "hazard_band": hazard_band}


def _frame_walls(maze, obj_color, floor, play_top):
    """Walls = the STRUCTURAL colour (the non-floor, non-object colour with the most cells that forms
    MULTIPLE connected components -- excludes the single-CC editor-panel border). Decomposed into
    horizontal ROW-RUNS (not bounding boxes) so a concave wall's interior PASSAGE is preserved -- a
    bbox would fill the gap and over-block the planner (the L7 failure mode)."""
    import scipy.ndimage as ndi

    best_cells, wall_color = -1, None
    for col in {int(c) for c in np.unique(maze)} - floor:
        if col == obj_color:
            continue
        cells = int((maze == col).sum())
        _, n = ndi.label(maze == col)
        if n >= 2 and cells > best_cells:  # structural = multi-CC, most cells
            best_cells, wall_color = cells, col
    boxes = []
    if wall_color is not None:
        mask = maze == wall_color
        for ry in range(mask.shape[0]):
            x = 0
            row = mask[ry]
            while x < len(row):
                if row[x]:
                    x0 = x
                    while x < len(row) and row[x]:
                        x += 1
                    boxes.append((x0, ry + play_top, x - x0, 1))  # one box per contiguous run
                else:
                    x += 1
    return boxes


def frame_to_maze_model(
    frame, n_slots, move_codes, *, settle_code=0, invisible_slots=3, play_top=2, maze_bottom=32
):
    """FRAME-ONLY: assemble a complete arc_maze_planner.MazeModel from a single frame, zero internal
    state. Geometry (object start, target, walls, checkpoints, hazard band + residual hidden hitboxes)
    is frame-induced; the `move_codes` (direction -> command code) + `invisible_slots` cadence come
    from the offline program-editor transition model (atomic-run, not frame-inducible). Returns a
    MazeModel ready for checkpoint_multirun_plan / timed_trap_plan, or None if the object/target
    cannot be read."""
    from carnot.agentic.arc_maze_planner import MazeModel

    g = np.asarray(frame.frame if hasattr(frame, "frame") else frame)
    if g.ndim == 3:
        g = g[-1]
    maze = g[play_top:maze_bottom]
    vals, counts = np.unique(maze, return_counts=True)
    floor = {int(c) for _, c in sorted(zip(counts.tolist(), vals.tolist()), reverse=True)[:3]}
    ot = induce_object_target_attrs(g, play_top=play_top, play_rows=maze_bottom)
    obj, tgt = ot["object"], ot["target"]
    if obj is None or tgt is None:
        return None
    sub = induce_maze_sub_fields(g, play_top=play_top, maze_bottom=maze_bottom)
    walls = _frame_walls(maze, ot["object_color"], floor, play_top)
    spikes_visible, spikes_hidden = [], []
    if sub["hazard_band"] is not None:
        hx, hy, hw, hh = sub["hazard_band"]
        spikes_visible = [sub["hazard_band"]]
        # the residual hidden hitboxes sit at the band's left + right edges (the marker concentrates
        # there); validated == tn36 internal hidden boxes (37,16,4,4)+(57,16,4,4) on L7.
        spikes_hidden = [(hx, hy, 4, hh), (hx + hw - 4, hy, 4, hh)]
    return MazeModel(
        object_wh=(4, 4),
        start=(obj.x, obj.y),
        target=(tgt.x, tgt.y),
        walls=tuple(walls),
        checkpoints=[(c[0], c[1]) for c in sub["checkpoints"]],
        move_codes=move_codes,
        settle_code=settle_code,
        n_slots=n_slots,
        bounds=64,
        spikes_visible=spikes_visible,
        spikes_hidden=spikes_hidden,
        invisible_slots=invisible_slots,
    )


def main() -> int:
    arc = kit.offline_arcade()
    game = sys.argv[1] if len(sys.argv) > 1 else "tn36"
    print(f"== FRAME-ONLY induction probe of {game} (no internal state) ==", flush=True)
    effects = probe(arc, game, step=1)
    out = induce(effects)
    print(f"DETECTED MECHANIC: {out['mechanic']}")
    # Wire the frame-only verdict into the STRATEGY ROUTER (the live path: an unseen game's detected
    # mechanic routes to a solving strategy with NO internal state / registry lookup).
    from carnot.agentic import arc_strategy_router as strat  # noqa: E402

    routed = strat.route_strategy(
        out["mechanic"] if out["mechanic"] != "unknown" else "graph_explore"
    )
    print(
        f"  STRATEGY ROUTE (frame-only): {routed['name']} (wired={routed['wired']}) — {routed['reason']}"
    )
    print(f"    solver: {routed['solver']}")
    print(f"  HUD cells (masked): {out['hud_cells']}")
    print(
        f"  edit-button palette: {out['n_edit_buttons']} toggle-cells, bbox {out['editor_bbox']} "
        f"({out['editor_block_wh'][0]}x{out['editor_block_wh'][1]}, density {out['editor_density']})"
    )
    sprites = play_area_sprites(arc, game, set(out["hud_cells"]), out["editor_bbox"])
    print(f"  play-area sprites (object + target candidates): {sprites}")

    # VALIDATION (separate; NOT used by the detector) -- compare to the internal layout.
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    env.reset()
    try:
        bz = env._game.fdksqlmpki.bzirenxmrg
        true_x = sorted({int(s.x) for s in bz.vupcwzjtxu.pfyayhyovw})
        o, t = bz.htntnzkbzu, bz.aqszntqeae
        print(
            f"  [validate] true editor slot-x span = {true_x[0]}..{true_x[-1]} "
            f"(detector bbox x {out['editor_bbox'][0]}..{out['editor_bbox'][2]})"
        )
        print(
            f"  [validate] true object @({o.x},{o.y}) target @({t.x},{t.y}) "
            f"vs detected play-area sprites above"
        )
        ed = out["editor_bbox"]
        ok = (
            out["mechanic"] == "program_editor"
            and ed[0] <= true_x[0] + 3
            and ed[2] >= true_x[-1] - 3
        )
        print(f"  [validate] program-editor detected AND bbox covers the true slots: {ok}")
    except Exception as e:  # pragma: no cover
        print(f"  [validate] (internal check unavailable: {type(e).__name__})")

    if out["mechanic"] == "program_editor":
        print(
            "== make the editor USABLE frame-only (click->(slot,bit) mapping + run-trigger) ==",
            flush=True,
        )
        layout = induce_editor_layout(arc, game)
        print(
            f"  derived {len(layout['slot_glyph_x'])} slots at glyph-x {layout['slot_glyph_x']}; "
            f"bit-rows y {layout['bit_rows']}"
        )
        run = find_run_button(arc, game, layout)
        print(f"  derived RUN button: {run}")
        if run:
            lvl = frame_only_solve(arc, game, layout, run)
            print(
                f"  FRAME-ONLY SOLVE of the current level -> level {lvl} "
                f"{'(SOLVED, zero internal state)' if lvl >= 1 else '(no advance)'}"
            )
            print(
                "== GENERAL winner-discovery: blind program-space search (binary win signal only) ==",
                flush=True,
            )
            ws = frame_only_winner_search(arc, game, layout, run)
            print(
                f"  winner-search: found={ws['found']} via {ws['strategy']} in {ws['runs']} runs "
                f"(full product space = {ws['product_space']}); NO graded feedback -> blind, "
                f"exponential in program length (does not scale)."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
