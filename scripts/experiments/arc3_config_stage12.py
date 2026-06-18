"""Config target-induction research build -- STAGE 1 (taxonomy) + STAGE 2 (executor with ground-truth
target). Plan: docs/research-notes/arc-config-target-induction-scope-2026-06-18.md.

STAGE 1 (taxonomy): for each config game, get the GROUND-TRUTH target -- replay the banked solve to the
first level-up, capture the PRE-win grid, and take the editable region's colours there as the target T.
Classify: (a) visible reference (a static region elsewhere matches T's colour-multiset, shape-agnostic),
(b) legend/rule (small distinct static region), else (c) relational/unknown.

STAGE 2 (executor positive control): given the TRUE target T, can a toggle-to-match search solve the
game first-contact? best_first_search over deep-copied real envs; candidates = click each editable cell
(the real env applies the actual recolor); heuristic = cell-mismatch(editable now, T); is_goal = a real
level-up. This isolates 'can we EXECUTE given a target' from 'can we INDUCE the target' -- it must pass
before Stage 3 induction is worth building. Gate: >=3 of the config games solve when handed the true T.

Zero quota, CPU, no LLM (Stage 3 adds the inducer). Uses the banked solve ONLY as ground truth for the
target (the executor still searches from reset; it does not replay the banked path)."""
from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash, objects
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed
from carnot.agentic.arc_world_model_dsl import _background
from carnot.agentic.arc_heuristic_search_over_verified_wm import best_first_search

CONFIG_GAMES = ["bp35", "dc22", "g50t", "ka59", "lf52", "s5i5", "tn36", "sc25", "tr87"]


def _mh():
    spec = importlib.util.spec_from_file_location(
        "mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def editable_mask(game, n=120):
    arc = kit.offline_arcade(); env = arc.make(game, scorecard_id=arc.open_scorecard()); f = env.reset()
    rng = random.Random(0); ch = None
    for _ in range(n):
        if f is None or np.asarray(grid_of(f)).size == 0:
            f = env.reset(); continue
        g = np.asarray(grid_of(f))
        if ch is None:
            ch = np.zeros_like(g, bool)
        av = list(getattr(f, "available_actions", []) or [])
        if 6 in av and objects(g):
            oy, ox = objects(g)[rng.randrange(len(objects(g)))]; a = 6; data = {"x": int(ox), "y": int(oy)}
        else:
            kb = [x for x in av if x not in (0, 6)] or [1, 2, 3, 4, 5]; a = rng.choice(kb); data = None
        nf = env.step(_game_action(GameAction, a), data=data)
        if nf is None:
            f = env.reset(); continue
        g1 = np.asarray(grid_of(nf))
        if g1.shape == g.shape:
            ch |= (g != g1)
        f = nf
    return ch


def banked_target(game, E, GameAction):
    """Replay the banked solve to the FIRST level-up; return (target_colors_on_E grid, pre_win_grid)."""
    mh = _mh()
    src = mh.RESOLVED_ARTIFACTS.get(game, mh.GAME_ARTIFACTS.get(game))
    if not src:
        return None                                    # no banked solve -> no ground-truth target
    acts = [mh.normalize(a) for a in (mh.load_actions(src) or []) if mh.normalize(a)[0] is not None]
    if not acts:
        return None
    warm = game in mh.WARMUP_GAMES
    arc = kit.offline_arcade(); env = arc.make(game, scorecard_id=arc.open_scorecard()); f = env.reset()
    if warm:
        aid, data = acts[0]; f = env.step(_game_action(GameAction, aid), data=data)
    for aid, data in acts:
        prev = np.asarray(grid_of(f))
        nf = env.step(_game_action(GameAction, aid), data=data)
        if nf is None:
            return None
        if _levels_completed(nf) > _levels_completed(f):
            return prev                                # the config that triggers the win
        f = nf
    return None


def classify_target(T, prewin, E):
    """(a) visible reference: a static (non-editable) region's colour-multiset matches T's; (b) legend:
    a small distinct static region; else (c) relational/unknown."""
    p = np.asarray(prewin); bg = _background(p)
    tcolors = sorted(int(p[r, c]) for r, c in np.argwhere(E))
    from collections import Counter
    tmulti = Counter(c for c in tcolors if c != bg)
    if not tmulti:
        return "c_relational_or_empty"
    # static non-bg cells (not editable)
    static = (p != bg) & ~E
    smulti = Counter(int(v) for v in p[static])
    # (a): every target colour appears in the static region with >= its target count (a reference exists)
    if smulti and all(smulti.get(col, 0) >= cnt for col, cnt in tmulti.items()):
        return "a_visible_reference"
    n_static = int(static.sum())
    if 0 < n_static <= 40:
        return "b_legend_small_static"
    return "c_relational_or_unknown"


def executor_solve(game, E, T_grid, GameAction, max_expansions=4000):
    """STAGE 2: best_first_search clicking editable cells, heuristic = mismatch(editable now, T), toward
    a real level-up. The real env applies the recolor; we only supply the click candidates + target."""
    import copy as _copy
    Ecells = [(int(r), int(c)) for r, c in np.argwhere(E)]
    T = np.asarray(T_grid)

    def mismatch(grid):
        g = np.asarray(grid)
        if g.shape != T.shape:
            return float(len(Ecells))
        return float(sum(1 for r, c in Ecells if g[r, c] != T[r, c]))

    env0 = kit.offline_arcade().make(game, scorecard_id=kit.offline_arcade().open_scorecard())
    f0 = env0.reset()
    if f0 is None:
        return {"solved": False, "nodes": 0}
    start_level = _levels_completed(f0)
    reg = {}
    gh0 = frame_hash(np.asarray(grid_of(f0))); reg[gh0] = (env0, f0)
    start = {"gh": gh0, "h": mismatch(grid_of(f0)), "level": start_level}

    def is_goal(s):
        return s["level"] > start_level

    def heuristic(s):
        return float(s["h"])

    Eset = set(Ecells)

    def next_states(s):
        cur = reg.get(s["gh"])
        if cur is None:
            return []
        env, frame = cur
        g = np.asarray(grid_of(frame))
        # candidates = click OBJECT centroids that lie in the editable region (config games respond to
        # clicks on objects/components, not bare cells); prefer objects currently mismatching the target.
        objs = [(int(y), int(x)) for (y, x) in objects(g)]
        in_edit = [(y, x) for (y, x) in objs if (y, x) in Eset] or objs
        if g.shape == T.shape:
            in_edit.sort(key=lambda yx: 0 if g[yx[0], yx[1]] != T[yx[0], yx[1]] else 1)
        out = []
        for (y, x) in in_edit[:24]:                     # cap branching for tractability
            e2 = _copy.deepcopy(env)
            nf = e2.step(_game_action(GameAction, 6), data={"x": x, "y": y})
            if nf is None:
                continue
            ng = np.asarray(grid_of(nf))
            if ng.size == 0:
                continue
            gh = frame_hash(ng)
            if gh not in reg:
                reg[gh] = (e2, nf)
            out.append(((6, x, y), {"gh": gh, "h": mismatch(ng), "level": _levels_completed(nf)}))
        return out

    res = best_first_search(start, next_states=next_states, is_goal=is_goal,
                            heuristic=heuristic, max_expansions=max_expansions)
    return {"solved": res.solved, "actions": (len(res.actions) if res.solved else None),
            "nodes": res.nodes_expanded}


def main():
    print("== CONFIG Stage 1 (taxonomy) + Stage 2 (executor with ground-truth target) ==", flush=True)
    rows = []
    from collections import Counter
    taxo = Counter(); solved_given_target = []
    for g in CONFIG_GAMES:
        t0 = time.time()
        E = editable_mask(g)
        if E is None or not E.any():
            rows.append({"game": g, "verdict": "no_editable_region"}); continue
        prewin = banked_target(g, E, GameAction)
        if prewin is None:
            rows.append({"game": g, "editable_cells": int(E.sum()), "verdict": "no_banked_target"}); continue
        cls = classify_target(prewin, prewin, E)
        taxo[cls] += 1
        ex = executor_solve(g, E, prewin, GameAction)
        if ex["solved"]:
            solved_given_target.append(g)
        rows.append({"game": g, "editable_cells": int(E.sum()), "target_class": cls,
                     "executor_solved_given_true_target": ex["solved"], "executor_nodes": ex["nodes"],
                     "executor_actions": ex["actions"], "dur_s": round(time.time() - t0, 1)})
        print(f"  {g:5} editable={int(E.sum()):3} class={cls:24} executor_solved={ex['solved']} "
              f"nodes={ex['nodes']} [{rows[-1]['dur_s']:.0f}s]", flush=True)
    findable = taxo.get("a_visible_reference", 0) + taxo.get("b_legend_small_static", 0)
    out = {"experiment": "arc3_config_stage12", "config_games": CONFIG_GAMES,
           "stage1_taxonomy": dict(taxo), "stage1_findable_or_derivable": findable,
           "stage1_gate_pass": findable >= 4,
           "stage2_executor_solved_given_true_target": solved_given_target,
           "stage2_gate_pass": len(solved_given_target) >= 3,
           "per_game": rows,
           "honest_verdict": (f"complete_config_stage12_taxo_findable_{findable}_of_9_"
                              f"executor_solved_{len(solved_given_target)}_given_true_target"),
           "inference_substrate": "offline_arc_agi3_config_stage12_no_llm_cpu"}
    (REPO / "results" / "arc3_config_stage12.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  STAGE 1 taxonomy: {dict(taxo)} | findable/derivable={findable}/9 (gate>=4: {out['stage1_gate_pass']})", flush=True)
    print(f"  STAGE 2 executor solved given true target: {solved_given_target} (gate>=3: {out['stage2_gate_pass']})", flush=True)
    print(f"  -> {out['honest_verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
