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
from arcengine import GameAction  # noqa: E402

LOCAL_MAX = 8        # a click changing <= this many non-HUD cells is a "local toggle" (an edit button)
HUD_FRAC = 0.5       # a cell changed by >= this fraction of all clicks is action-invariant HUD


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
    is_editor = (len(edits) >= 20 and bbox is not None
                 and block_w <= 40 and block_h <= 40 and density >= 0.5)
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
    mask = (g0 != bg)
    mask[play_top:, :] = False                     # drop the editor + HUD rows
    lab, n = ndi.label(mask)
    sprites = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(xs) >= 3:
            sprites.append((int(xs.mean()), int(ys.mean()), int(len(xs))))
    return sorted(sprites, key=lambda s: -s[2])[:6]


def main() -> int:
    arc = kit.offline_arcade()
    game = sys.argv[1] if len(sys.argv) > 1 else "tn36"
    print(f"== FRAME-ONLY induction probe of {game} (no internal state) ==", flush=True)
    effects = probe(arc, game, step=1)
    out = induce(effects)
    print(f"DETECTED MECHANIC: {out['mechanic']}")
    print(f"  HUD cells (masked): {out['hud_cells']}")
    print(f"  edit-button palette: {out['n_edit_buttons']} toggle-cells, bbox {out['editor_bbox']} "
          f"({out['editor_block_wh'][0]}x{out['editor_block_wh'][1]}, density {out['editor_density']})")
    sprites = play_area_sprites(arc, game, set(out["hud_cells"]), out["editor_bbox"])
    print(f"  play-area sprites (object + target candidates): {sprites}")

    # VALIDATION (separate; NOT used by the detector) -- compare to the internal layout.
    env = arc.make(game, scorecard_id=arc.open_scorecard()); env.reset()
    try:
        bz = env._game.fdksqlmpki.bzirenxmrg
        true_x = sorted({int(s.x) for s in bz.vupcwzjtxu.pfyayhyovw})
        o, t = bz.htntnzkbzu, bz.aqszntqeae
        print(f"  [validate] true editor slot-x span = {true_x[0]}..{true_x[-1]} "
              f"(detector bbox x {out['editor_bbox'][0]}..{out['editor_bbox'][2]})")
        print(f"  [validate] true object @({o.x},{o.y}) target @({t.x},{t.y}) "
              f"vs detected play-area sprites above")
        ed = out["editor_bbox"]
        ok = (out["mechanic"] == "program_editor" and ed[0] <= true_x[0] + 3 and ed[2] >= true_x[-1] - 3)
        print(f"  [validate] program-editor detected AND bbox covers the true slots: {ok}")
    except Exception as e:  # pragma: no cover
        print(f"  [validate] (internal check unavailable: {type(e).__name__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
