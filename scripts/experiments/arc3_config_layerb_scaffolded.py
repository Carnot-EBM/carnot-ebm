"""Config rule-inducer LAYER B -- SCAFFOLDED LLM-scene-reader. Plan: arc-config-target-induction-scope.

Move #1 (arc3_config_layerb_llm.py) proved the local gemma-12B-Q4 CANNOT read a raw 64x64 ASCII scene:
it loses spatial track and degenerates into a comment-loop, on BOTH the hard rewrite (tr87) and the
simple recolor strip (ka59). The diagnosed lever: remove the grid-parsing burden via STRUCTURED
EXTRACTION -- hand the model small cropped regions, not 4096 cells of mostly-background.

This harness does that. It extracts:
  - the EDITABLE region (the cells the player changes), cropped to its bbox -> a small array;
  - the WIN editable values + several NON-win editable values as LABELLED examples (the discriminating
    signal -- tests whether the model can EXPRESS the rule, separate from guessing it blind);
  - the REFERENCE region (the static, non-background, non-editable 'rule area'), cropped to its bbox.
It then asks the model for a RELATIONAL is_win(grid) and explicitly warns against hardcoding the literal
win array (that memorizes, it does not generalize).

TIERED honest verdict (move #1 failed at Tier 0):
  Tier 0 reading-fixed : produces COHERENT parseable+runnable is_win (not degenerate comment-rambling).
  Tier 1 grounds       : is_win fires True on the banked win, False on the non-wins.
  Tier 2 relational     : grounds AND is not a literal-array hardcode (a real, generalizable rule).

Propose-then-ground: the verifier grounds the induced predicate against the banked win + non-wins.
iGPU port 8920 (offline-legal, ~4.2 tok/s, never the 3090s); zero quota; tears down its server.

Usage: python scripts/experiments/arc3_config_layerb_scaffolded.py [game]   (default ka59)."""
from __future__ import annotations

import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_agi3_world_model import grid_of, objects
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed
from carnot.agentic.arc_world_model_dsl import _background

# reuse the bounded-generation helper proven in move #1 (no ``` stop; raw capture; ~260s bound)
_lbspec = importlib.util.spec_from_file_location("lbmod", str(REPO / "scripts" / "experiments" / "arc3_config_layerb_llm.py"))
_lbmod = importlib.util.module_from_spec(_lbspec); _lbspec.loader.exec_module(_lbmod)
_generate_bounded = _lbmod._generate_bounded

GAME = sys.argv[1] if len(sys.argv) > 1 else "ka59"


def _mh():
    spec = importlib.util.spec_from_file_location(
        "mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _render(sub):
    return "\n".join("".join(str(int(v))[-1] for v in row) for row in np.asarray(sub))


def _explore_step(env, f, g, rng):
    av = list(getattr(f, "available_actions", []) or [])
    if 6 in av and objects(g):
        oy, ox = objects(g)[rng.randrange(len(objects(g)))]
        return env.step(_game_action(GameAction, 6), data={"x": int(ox), "y": int(oy)})
    kb = [x for x in av if x not in (0, 6)] or [1, 2, 3, 4]
    return env.step(_game_action(GameAction, rng.choice(kb)), data=None)


def collect():
    """Discover editable region; gather WIN + NON-win editable sub-arrays + the reference bbox sub-grid."""
    arc = kit.offline_arcade(); env = arc.make(GAME, scorecard_id=arc.open_scorecard()); f = env.reset()
    scene = np.asarray(grid_of(f)).copy(); bg = int(_background(scene))
    rng = random.Random(0); ch = np.zeros_like(scene, bool); nonwins = []
    g = scene
    for i in range(60):
        nf = _explore_step(env, f, g, rng)
        if nf is None:
            f = env.reset(); g = np.asarray(grid_of(f)); continue
        g1 = np.asarray(grid_of(nf))
        if g1.shape == g.shape:
            ch |= (g != g1)
            if i % 10 == 0 and _levels_completed(nf) == 0:
                nonwins.append(g1.copy())
        f = nf; g = g1
    if not ch.any():
        return scene, None, bg, None, None, None, []
    rs = np.argwhere(ch); eb = (int(rs[:, 0].min()), int(rs[:, 0].max()), int(rs[:, 1].min()), int(rs[:, 1].max()))
    # reference region = static (non-bg) cells NOT in the editable ROW(S) -> crop to its bbox. Excluding
    # the whole editable row (not just the editable cols) keeps UI/frame cells in the editable row out of
    # the reference, so the reference crops down to the actual 'rule area' and renders (move #1 v1 leaked
    # the editable row into ref_box, blowing it past the render cap).
    static = (scene != bg).copy()
    static[eb[0]:eb[1] + 1, :] = False
    ref_box = None
    if static.any():
        sr = np.argwhere(static)
        ref_box = (int(sr[:, 0].min()), int(sr[:, 0].max()), int(sr[:, 1].min()), int(sr[:, 1].max()))
    # banked win config
    mh = _mh(); src = mh.RESOLVED_ARTIFACTS.get(GAME, mh.GAME_ARTIFACTS.get(GAME))
    acts = [mh.normalize(a) for a in (mh.load_actions(src) or []) if mh.normalize(a)[0] is not None]
    arc2 = kit.offline_arcade(); env2 = arc2.make(GAME, scorecard_id=arc2.open_scorecard()); f2 = env2.reset()
    win = None
    for aid, data in acts:
        prev = np.asarray(grid_of(f2)); f2 = env2.step(_game_action(GameAction, aid), data=data)
        if f2 is None:
            break
        if _levels_completed(f2) > 0:
            win = prev; break
    return scene, eb, bg, ref_box, win, ch, nonwins


def _edit_sub(grid, eb):
    g = np.asarray(grid)
    return g[eb[0]:eb[1] + 1, eb[2]:eb[3] + 1]


def _summ(sub):
    """Exact, pre-computed description of an editable sub-array so the model never counts ASCII by eye:
    flat value list + per-colour counts."""
    from collections import Counter
    flat = [int(v) for v in np.asarray(sub).flatten()]
    counts = dict(sorted(Counter(flat).items()))
    return f"values={flat}  counts={counts}"


def _digest_reference(scene, bg, ref_box):
    """Digest the static reference region into COMPACT object-centric features (per-colour connected
    components: count, n_components, bounding boxes) -- NEVER a raw grid. Handing the model a raw grid
    (even a cropped 21x45) re-triggers the reading-degeneration that move #1 diagnosed; the model narrates
    the grid cell-by-cell and never writes the predicate. Component features keep the perception burden
    off the model (the same object-centric representation that is the pipeline's load-bearing finding)."""
    from carnot.agentic.arc_world_model_dsl import _color_components
    if ref_box is None:
        return "(no reference region found)"
    r0, r1, c0, c1 = ref_box
    sub = np.asarray(scene)[r0:r1 + 1, c0:c1 + 1]
    lines = []
    for col in sorted(set(int(v) for v in sub.flatten()) - {bg}):
        comps = _color_components(sub, col)
        boxes = []
        for comp in comps[:6]:
            ys = [p[0] for p in comp]; xs = [p[1] for p in comp]
            boxes.append(f"(r{min(ys)}-{max(ys)},c{min(xs)}-{max(xs)},n{len(comp)})")
        lines.append(f"  colour {col}: total_cells={int((sub == col).sum())}, components={len(comps)} {' '.join(boxes)}")
    hdr = f"reference region rows {r0}..{r1}, cols {c0}..{c1} (component coords are RELATIVE to r{r0},c{c0}):"
    return hdr + "\n" + "\n".join(lines)


def build_prompt(scene, eb, bg, ref_box, win, nonwins):
    win_sub = _edit_sub(win, eb)
    nonwin_strs = "\n".join(f"  NON-WIN {i + 1}: {_summ(_edit_sub(nw, eb))}" for i, nw in enumerate(nonwins[:5]))
    ref_str = _digest_reference(scene, bg, ref_box)
    return f"""You are inducing the WIN RULE of an ARC-AGI-3 configuration puzzle ('{GAME}'). I have already
done the hard perception work for you -- you do NOT need to read a full 64x64 grid. Reason ONLY over the
small extracted regions below.

The player edits the EDITABLE region: rows {eb[0]}..{eb[1]}, cols {eb[2]}..{eb[3]} (background colour={bg}).
The level COMPLETES when the editable region's values satisfy a rule defined by the static REFERENCE
region elsewhere in the scene.

EDITABLE region in the WINNING configuration (exact values + per-colour counts; do NOT re-count by eye):
  WIN: {_summ(win_sub)}

EDITABLE region in NON-winning configurations:
{nonwin_strs}

The static REFERENCE region (the 'rule area'); it does not change as the player edits:
{ref_str}

Write a Python predicate that recomputes the rule and returns True ONLY for winning configurations:

    import numpy as np
    def is_win(grid):
        # grid: 64x64 numpy int array.
        e = grid[{eb[0]}:{eb[1] + 1}, {eb[2]}:{eb[3] + 1}]   # the editable region
        # Return True iff e satisfies the win rule implied by the reference region.
        ...

Rules:
- Derive a RELATION between the editable region and the reference region (e.g. e equals / mirrors /
  encodes / counts the reference). Do NOT hardcode the literal winning array -- that will not generalize
  to other levels and is wrong.
- Use ONLY numpy + stdlib. Make is_win deterministic. Start immediately with `import numpy as np`.

Return ONLY one ```python code block with def is_win.
```python
"""


def verify(is_win, win, nonwins):
    res = {"fires_on_win": None, "false_positive_rate": None, "n_nonwin": len(nonwins)}
    try:
        res["fires_on_win"] = bool(is_win(np.asarray(win))) if win is not None else None
    except Exception as ex:
        res["win_error"] = f"{type(ex).__name__}: {str(ex)[:100]}"
    fp = 0
    for nw in nonwins:
        try:
            if bool(is_win(np.asarray(nw))):
                fp += 1
        except Exception:
            pass
    res["false_positive_rate"] = round(fp / max(1, len(nonwins)), 3)
    return res


def _strip_comments(code):
    """Drop full-line and trailing # comments so the literal-hardcode check inspects EXECUTABLE code
    only -- the model echoes the win array in comments, which falsely tripped the detector."""
    out = []
    for line in code.splitlines():
        out.append(re.sub(r"#.*$", "", line))
    return "\n".join(out)


def _looks_literal_hardcode(code, win_sub):
    """Heuristic: does the EXECUTABLE predicate embed the literal winning array (memorization, not a
    rule)? Flag if a long run of the win array's values appears as a literal in the code (comments
    stripped first). A derived constant (e.g. count_4 == 32, where 32 was read from the reference) is
    NOT a literal-array hardcode -- it is a relational rule with the relation pre-evaluated."""
    vals = [int(v) for v in np.asarray(win_sub).flatten()]
    if len(vals) < 6:
        return False
    flat = re.sub(r"\s+", "", _strip_comments(code))
    bare = "".join(str(v) for v in vals)
    if bare in flat:
        return True
    return any("".join(str(v) for v in vals[i:i + 8]) in flat for i in range(0, max(1, len(vals) - 8)))


def main():
    print(f"== SCAFFOLDED LAYER B on {GAME} (structured extraction vs raw-scene move #1) ==", flush=True)
    scene, eb, bg, ref_box, win, ch, nonwins = collect()
    if eb is None or win is None or len(nonwins) < 2:
        out = {"experiment": "arc3_config_layerb_scaffolded", "game": GAME,
               "honest_verdict": "complete_scaffolded_blocked_no_editable_or_no_win_or_too_few_nonwins",
               "inference_substrate": "offline_arc_agi3_layerb_scaffolded_iGPU_port8920"}
        (REPO / "results" / f"arc3_config_layerb_scaffolded_{GAME}.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"  -> {out['honest_verdict']}", flush=True); return 0
    win_sub = _edit_sub(win, eb)
    print(f"  editable bbox={eb} ({win_sub.size} cells) | ref_box={ref_box} | non-wins={len(nonwins)}", flush=True)
    proposer = e3.LocalGGUFProposer(repo_substr="gemma-4-12B-it", port=8920, timeout=600)
    out = {"experiment": "arc3_config_layerb_scaffolded", "game": GAME,
           "editable_bbox": list(eb), "editable_cells": int(win_sub.size), "reference_bbox": list(ref_box) if ref_box else None,
           "n_nonwins": len(nonwins), "generation": {"n_predict": 1100, "bounded": True, "scaffolded": True},
           "inference_substrate": "offline_arc_agi3_layerb_scaffolded_iGPU_port8920"}
    try:
        ok, code, raw = _generate_bounded(proposer, build_prompt(scene, eb, bg, ref_box, win, nonwins))
        out["raw_sample"] = str(raw)[:800]
        out["coherent_runnable"] = bool(ok)  # Tier 0: parseable + has def is_win
        if not ok:
            out["msg"] = str(code)[:200]
            out["honest_verdict"] = "complete_scaffolded_tier0_fail_no_coherent_predicate"
        else:
            (REPO / "results" / "arc_config_layerb").mkdir(parents=True, exist_ok=True)
            (REPO / "results" / "arc_config_layerb" / f"{GAME}_scaffolded_is_win.py").write_text(code)
            ns = {}
            try:
                exec(code, ns)  # noqa: S102 -- inducing a verifier predicate; sandboxed-by-scope
                v = verify(ns.get("is_win", lambda g: False), win, nonwins)
                out["verification"] = v
                fpr = v.get("false_positive_rate")  # 0.0 is the BEST value -- do NOT use `or` (falsy-zero)
                grounded = bool(v.get("fires_on_win")) and fpr is not None and fpr < 0.2
                literal = _looks_literal_hardcode(code, win_sub)
                out["rule_grounded"] = grounded
                out["literal_hardcode"] = bool(literal)
                if grounded and not literal:
                    out["honest_verdict"] = "complete_scaffolded_tier2_GROUNDED_relational_rule"
                elif grounded:
                    out["honest_verdict"] = "complete_scaffolded_tier1_grounded_but_literal_hardcode_not_generalizable"
                else:
                    out["honest_verdict"] = "complete_scaffolded_tier0_coherent_but_rule_not_grounded"
            except Exception as ex:
                out["exec_error"] = f"{type(ex).__name__}: {str(ex)[:120]}"
                out["honest_verdict"] = "complete_scaffolded_tier0_coherent_but_uncompilable_at_runtime"
    finally:
        try:
            proposer.stop()
        except Exception:
            pass
    (REPO / "results" / f"arc3_config_layerb_scaffolded_{GAME}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"  coherent={out.get('coherent_runnable')} verification={out.get('verification')} "
          f"grounded={out.get('rule_grounded')} literal={out.get('literal_hardcode')}", flush=True)
    print(f"  -> {out.get('honest_verdict')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
