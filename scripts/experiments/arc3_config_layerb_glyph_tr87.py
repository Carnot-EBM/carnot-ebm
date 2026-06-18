"""Config Layer B -- REWRITE class (tr87): a from-PIXELS glyph-substitution verifier.

Move-2 (scaffolded LLM induction) grounded ka59 (count class) but only reached Tier 0 on tr87, because
tr87's rule is a GLYPH SUBSTITUTION over sprite bitmaps, not a cell-value relation. Reframing: tr87's
mechanic is KNOWN (the adapter RE'd it 2026-06-17), so its Layer-B verifier is a DETERMINISTIC glyph
decode + rewrite check, NOT an LLM induction (LLM induction is for UNKNOWN mechanics like ka59). The
adapter read the game's INTERNAL state and explicitly deferred "classifying glyph bitmaps + decoding the
rule grid from pixels" as a future upgrade -- this script IS that upgrade.

tr87 pixel structure (RE'd here, win frame):
  - glyph VALUE = the 5-on bitmap pattern of a 5x5 sprite tile (frame-colour-agnostic);
  - A-series glyphs (the TARGET row + rule LHS) are framed with colour 10;
  - B-series glyphs (the EDITABLE row + rule RHS) are framed with colour 7;
  - 3 rule bands pair (A-glyph, B-glyph) = the substitution map A_value -> B_value;
  - WIN: editable_Bvalues == [map[a] for a in target_Avalues].

The verifier GROUNDS if it fires True on the banked win and False on the non-wins -- the same
propose-then-ground bar, but the "proposer" is the RE'd mechanic, not the LLM. Reusable primitive:
`segment_glyphs` (split a sprite band into 5x5 tiles by on-pixel column gaps) + `value_id` (bitmap
identity, frame-agnostic) -- a sprite-glyph perception primitive for any glyph game, the rewrite-class
analogue of the connected-component digest that grounded ka59. No LLM, no internal state, CPU, zero quota.

Usage: python scripts/experiments/arc3_config_layerb_glyph_tr87.py"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed

GAME = "tr87"
ON = 5  # the "on" pixel colour shared by every glyph regardless of A/B frame colour (10 vs 7)


def _value_registry():
    reg: dict = {}

    def value_id(tile):
        """Frame-agnostic glyph value = the 5-on bitmap pattern, CROPPED to its on-pixel bounding box so a
        glyph clusters identically regardless of surrounding frame width (variable-width segmentation
        otherwise gives the same glyph different bitmaps across regions). Same cropped on-pattern -> same
        value id, whether A-framed (10) or B-framed (7)."""
        on = (np.asarray(tile) == ON)
        if not on.any():
            key = ()
        else:
            rs = np.argwhere(on)
            r0, r1, c0, c1 = rs[:, 0].min(), rs[:, 0].max(), rs[:, 1].min(), rs[:, 1].max()
            key = tuple(map(tuple, on[r0:r1 + 1, c0:c1 + 1].astype(int)))
        if key not in reg:
            reg[key] = len(reg)
        return reg[key]

    return value_id


def _content_rows(g, r0, r1, c0, c1):
    """Rows in [r0,r1] that contain at least one ON pixel within cols [c0,c1] (the glyph interior rows)."""
    return [r for r in range(r0, r1 + 1) if (np.asarray(g)[r, c0:c1 + 1] == ON).any()]


def segment_glyphs(g, rows, c0=0, c1=63):
    """Split a sprite band (given its content `rows`) into glyph tiles by ON-pixel column gaps: a column
    is a SEPARATOR if no row in the band has an ON pixel there; maximal runs of non-separator columns are
    glyph tiles. Returns list of (col_start, col_end_inclusive, tile_array, frame_colour)."""
    g = np.asarray(g)
    if not rows:
        return []
    r0, r1 = min(rows), max(rows)
    has_on = [(g[r0:r1 + 1, c] == ON).any() for c in range(c0, c1 + 1)]
    tiles, start = [], None
    for j, on in enumerate(has_on + [False]):
        c = c0 + j
        if on and start is None:
            start = c
        elif not on and start is not None:
            tile = g[r0:r1 + 1, start:c]
            # frame colour = the non-ON, non-background colour most present around the tile (10 -> A, 7 -> B)
            vals = [int(v) for v in tile.flatten() if int(v) in (7, 10)]
            frame = 10 if vals.count(10) >= vals.count(7) else 7
            tiles.append((start, c - 1, tile, frame))
            start = None
    return tiles


def collect():
    arc = kit.offline_arcade(); env = arc.make(GAME, scorecard_id=arc.open_scorecard()); f = env.reset()
    # banked win + a few non-wins (replay the banked solve; pre-win grids before the level-up are wins-in-waiting)
    spec = importlib.util.spec_from_file_location("mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    mh = importlib.util.module_from_spec(spec); spec.loader.exec_module(mh)
    src = mh.RESOLVED_ARTIFACTS.get(GAME, mh.GAME_ARTIFACTS.get(GAME))
    acts = [mh.normalize(a) for a in (mh.load_actions(src) or []) if mh.normalize(a)[0] is not None]
    env2 = kit.offline_arcade().make(GAME, scorecard_id=kit.offline_arcade().open_scorecard()); f2 = env2.reset()
    win = None; nonwins = []
    for i, (aid, data) in enumerate(acts):
        prev = np.asarray(grid_of(f2)); f2 = env2.step(_game_action(GameAction, aid), data=data)
        if f2 is None:
            break
        if _levels_completed(f2) > 0:
            win = prev; break
        if i % 3 == 0:
            nonwins.append(prev.copy())
    return win, nonwins[-6:]


def decode_and_check(grid):
    """Decode glyph values from pixels and return True iff editable == rewrite(target) under the rule map.
    Regions are the RE'd tr87 layout (win frame): rule bands rows 5-9/14-18/23-27, target rows 41-45,
    editable rows 52-56."""
    g = np.asarray(grid)
    value_id = _value_registry()  # shared id space across ALL regions (A & B glyphs cluster by on-pattern)

    def ids(rows, c0=0, c1=63):
        tiles = segment_glyphs(g, _content_rows(g, rows[0], rows[1], c0, c1), c0, c1)
        return [(value_id(t), frame) for (_, _, t, frame) in tiles]

    # rule map: in each rule band, glyphs alternate A(frame10) then B(frame7); pair consecutive A->B
    rule_map = {}
    for band in ((5, 9), (14, 18), (23, 27)):
        seq = ids(band)
        a = None
        for vid, frame in seq:
            if frame == 10:
                a = vid
            elif frame == 7 and a is not None:
                rule_map[a] = vid; a = None
    target = [vid for vid, frame in ids((41, 45))]        # A-series target row
    editable = [vid for vid, frame in ids((52, 56))]       # B-series editable row
    if not rule_map or not target or not editable:
        return False, {"rule_map": len(rule_map), "target": len(target), "editable": len(editable)}
    expanded = [rule_map.get(a) for a in target]
    ok = (None not in expanded) and (expanded == editable)
    return ok, {"rule_map_size": len(rule_map), "target": target, "editable": editable, "expanded": expanded}


def main():
    print(f"== GLYPH-substitution from-PIXELS verifier on {GAME} (rewrite class) ==", flush=True)
    win, nonwins = collect()
    out = {"experiment": "arc3_config_layerb_glyph_tr87", "game": GAME,
           "inference_substrate": "offline_arc_agi3_glyph_substitution_pixel_decode_cpu_no_llm"}
    if win is None:
        out["honest_verdict"] = "complete_glyph_blocked_no_banked_win"
        (REPO / "results" / f"arc3_config_layerb_glyph_{GAME}.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"  -> {out['honest_verdict']}", flush=True); return 0
    fires_win, win_dbg = decode_and_check(win)
    fp = 0; nonwin_dbg = []
    for nw in nonwins:
        f_nw, dbg = decode_and_check(nw)
        nonwin_dbg.append(f_nw)
        if f_nw:
            fp += 1
    fpr = round(fp / max(1, len(nonwins)), 3)
    grounded = bool(fires_win) and fpr < 0.2
    out.update({"fires_on_win": bool(fires_win), "false_positive_rate": fpr, "n_nonwin": len(nonwins),
                "win_decode": win_dbg, "nonwin_fires": nonwin_dbg,
                "rule_grounded": grounded,
                "honest_verdict": ("complete_glyph_pixel_verifier_GROUNDED_rewrite_class" if grounded else
                                   "complete_glyph_pixel_decode_partial_not_grounded")})
    (REPO / "results" / f"arc3_config_layerb_glyph_{GAME}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"  fires_on_win={fires_win} fpr={fpr} grounded={grounded}", flush=True)
    print(f"  win decode: target={win_dbg.get('target')} editable={win_dbg.get('editable')} expanded={win_dbg.get('expanded')}", flush=True)
    print(f"  -> {out['honest_verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
