"""MECHANIC-PRIMITIVE TRANSFER PROBE -- the first real test of the offline->live bridge mechanism.

The whole-engine LOGO probe ruled out reusing an engine across games (0 transfer, even ar25->ka59
which share the 'move a selected object to a goal' mechanic): an induced engine hardcodes the
game's specific object/layout/colors. The live-viable bridge is instead a PARAMETERIZED MECHANIC
PRIMITIVE -- the shared STRUCTURE (a directional action translates a movable object by one step) --
whose game-specific PARAMETERS (which color is movable, which action -> which direction) are RE-FIT
to the new game's own observed transitions at test time.

This probe implements `MoveObjectPrimitive` (the structure abstracted from the ar25/ka59 mechanic)
and measures, for each game:

  prim_fit   : fit the primitive's params on FEW observed transitions (the live setting: you see a
               handful of moves before you must act), evaluate on a HELD-OUT test set. Metric =
               changed-cell accuracy (of the cells that actually moved, how many does it predict
               right) + exact-grid match.
  llm_engine : the existing LLM-induced world_model.py engine on the SAME test set -- the
               'induce from scratch' comparator.
  learning curve: prim accuracy vs #fit-transitions {3,5,10,20} -- does a FEW live observations
               suffice to re-parameterize the primitive?

TRANSFER is demonstrated if ONE primitive STRUCTURE, with per-game params fit from few observations,
models BOTH games of the similar pair well AND beats the from-scratch LLM engine -- i.e. the
abstraction generalizes, and live you only fit cheap params instead of inducing a whole engine.
Proposer-free, zero quota, CPU.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_action

GAMES = ["ar25", "ka59", "cn04", "sc25", "tu93", "sp80", "cd82", "m0r0", "sk48"]
# ar25/ka59 = the operator's "similar solves" pair (both manipulate a selected object to a goal);
# tu93/sp80/cd82/m0r0/sk48 = directional-movement candidates (the games most likely to be pure
# TRANSLATION at the dynamics level -- the real test of whether ANY game is a clean primitive fit).
EXPLORE_N = 120
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1), "0": (0, 0)}


# --------------------------------------------------------------------------- transitions
def _mh():
    spec = importlib.util.spec_from_file_location(
        "mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def banked_transitions(game, cell):
    mh = _mh()
    src = mh.RESOLVED_ARTIFACTS.get(game, mh.GAME_ARTIFACTS.get(game))
    if not src:
        return []
    acts = [mh.normalize(a) for a in (mh.load_actions(src) or []) if mh.normalize(a)[0] is not None]
    if not acts:
        return []
    warm = game in mh.WARMUP_GAMES
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env.reset()
    if warm:
        aid, data = acts[0]
        f = env.step(_game_action(GameAction, aid), data=data)
    out = []
    for aid, data in acts:
        g0 = e3.to_logical(grid_of(f), cell)
        l0 = _levels_completed(f)
        nf = env.step(_game_action(GameAction, aid), data=data)
        if nf is None:
            break
        out.append(e3.Transition(g0, int(aid), data, e3.to_logical(grid_of(nf), cell),
                                 l0, _levels_completed(nf)))
        f = nf
    return out


def collect_changed(game):
    """Frame-only transitions where the grid changed AND no level-up (real dynamics, no win discontinuity)."""
    explore, cell = e3.collect_transitions(game, n=EXPLORE_N)
    allt = explore + banked_transitions(game, cell)
    return [t for t in allt
            if t.level_after == t.level_before and not np.array_equal(t.grid, t.next_grid)], cell


# --------------------------------------------------------------------------- the primitive
def _infer_move(grid, next_grid, background):
    """Infer (movable_color, (dr,dc)) for a single transition: find a non-background color whose
    cell-set RIGIDLY TRANSLATES (same count, and aligning the two masks' top-left bounding-box
    corners maps one set exactly onto the other). Bounding-box-corner alignment is exact for a
    rigid translation regardless of overlap (unlike a centroid/edge heuristic), and the set-equality
    VERIFY self-rejects non-translations (reflection, push-with-obstacle, redraw). Displacement-
    agnostic + pixel-resolution-robust. Returns None if no color cleanly translates."""
    best = None
    for c in (set(int(v) for v in np.unique(grid)) - {background}):
        old = np.argwhere(grid == c)
        new = np.argwhere(next_grid == c)
        if len(old) < 2 or len(old) != len(new):
            continue
        dr = int(new[:, 0].min() - old[:, 0].min())     # align top-left corners of the two masks
        dc = int(new[:, 1].min() - old[:, 1].min())
        if dr == 0 and dc == 0:
            continue
        if {(r + dr, cc + dc) for r, cc in old} == set(map(tuple, new)):   # exact rigid translation
            # prefer the larger translating object (the agent's piece, not a 1px HUD tick)
            if best is None or len(old) > best[2]:
                best = (c, (dr, dc), len(old))
    return (best[0], best[1]) if best else None


class MoveObjectPrimitive:
    """Parameterized shared mechanic: a directional ACTION translates the MOVABLE-colored object by
    one step. Params (game-specific, fit from observed transitions): movable_color, background,
    direction_map {action_id -> (dr,dc)}. STRUCTURE is shared across games; only params are re-fit."""

    def __init__(self):
        self.movable = None
        self.background = 0
        self.direction_map = {}
        self.fitted = False

    def fit(self, transitions):
        """Infer movable_color + per-action direction from (grid, action, next_grid) examples."""
        # background = globally most common cell value
        allvals = Counter()
        for t in transitions:
            allvals.update(t.grid.flatten().tolist())
        self.background = allvals.most_common(1)[0][0] if allvals else 0
        # for each transition, find the color that translated, and by what (dr,dc) under that action
        votes_color = Counter()
        act_dir = {}   # action -> Counter of (dr,dc)
        for t in transitions:
            mv = _infer_move(t.grid, t.next_grid, self.background)
            if mv is not None:
                color, sh = mv
                votes_color[color] += 1
                act_dir.setdefault(t.action, Counter())[sh] += 1
        if not votes_color:
            self.fitted = False
            return self
        self.movable = votes_color.most_common(1)[0][0]
        self.direction_map = {a: cnt.most_common(1)[0][0] for a, cnt in act_dir.items()}
        self.fitted = True
        return self

    def engine(self, grid, action, data):
        if not self.fitted or action not in self.direction_map:
            return grid
        dr, dc = self.direction_map[action]
        m = np.argwhere(grid == self.movable)
        if len(m) == 0:
            return grid
        shifted = m + np.array([dr, dc])
        h, w = grid.shape
        if shifted[:, 0].min() < 0 or shifted[:, 0].max() >= h or \
           shifted[:, 1].min() < 0 or shifted[:, 1].max() >= w:
            return grid                       # move would leave the board -> blocked (no-op)
        out = grid.copy()
        for r, c in m:
            out[r, c] = self.background
        for r, c in shifted:
            out[r, c] = self.movable
        return out


# --------------------------------------------------------------------------- scoring
def _load_llm_engine(game):
    p = REPO / "results" / "arc_e3" / game / "world_model.py"
    if not p.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(f"llm_wm_{game}", str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "engine", None)
    except Exception:
        return None


def changed_cell_acc(engine, transitions):
    """Mean over transitions of: of the cells that ACTUALLY changed, what fraction does engine predict
    right. Plus exact-grid-match rate. The harsh+graded pair."""
    if not transitions:
        return float("nan"), float("nan")
    accs, exacts = [], 0
    for t in transitions:
        try:
            pred = np.asarray(engine(t.grid.copy(), t.action, t.data))
        except Exception:
            accs.append(0.0)
            continue
        if pred.shape != t.next_grid.shape:
            accs.append(0.0)
            continue
        mask = t.grid != t.next_grid
        accs.append(float((pred[mask] == t.next_grid[mask]).mean()) if mask.any() else 1.0)
        exacts += int(np.array_equal(pred, t.next_grid))
    return round(float(np.mean(accs)), 3), round(exacts / len(transitions), 3)


def probe_game(game):
    changed, cell = collect_changed(game)
    rec = {"game": game, "n_changed": len(changed), "cell": cell}
    if len(changed) < 8:
        rec["verdict"] = "INSUFFICIENT_DYNAMICS_TRANSITIONS"
        return rec
    # deterministic split: first 60% fit pool (we draw few from it), last 40% test
    split = max(4, int(len(changed) * 0.6))
    fit_pool, test = changed[:split], changed[split:]
    # learning curve: few observed transitions suffice to re-parameterize?
    curve = []
    for k in (3, 5, 10, 20):
        if k > len(fit_pool):
            break
        prim = MoveObjectPrimitive().fit(fit_pool[:k])
        cacc, exact = changed_cell_acc(prim.engine, test)
        curve.append({"k_fit": k, "movable": prim.movable, "n_actions_mapped": len(prim.direction_map),
                      "changed_cell_acc": cacc, "exact_match": exact})
    rec["primitive_learning_curve"] = curve
    rec["primitive_best"] = max(curve, key=lambda r: r["changed_cell_acc"]) if curve else None
    # from-scratch comparator: the LLM-induced engine on the SAME test set
    llm = _load_llm_engine(game)
    if llm is not None:
        cacc, exact = changed_cell_acc(llm, test)
        rec["llm_engine"] = {"changed_cell_acc": cacc, "exact_match": exact}
    else:
        rec["llm_engine"] = None
    pb = rec["primitive_best"]
    le = rec["llm_engine"]
    rec["primitive_beats_llm"] = bool(pb and le and pb["changed_cell_acc"] > (le["changed_cell_acc"] or 0) + 0.05)
    return rec


def main() -> int:
    print(f"== MECHANIC-PRIMITIVE TRANSFER probe (games={GAMES}; similar pair ar25<->ka59) ==", flush=True)
    rows = [probe_game(g) for g in GAMES]
    for r in rows:
        if r.get("primitive_best"):
            pb, le = r["primitive_best"], r.get("llm_engine")
            print(f"\n  [{r['game']}] n_changed={r['n_changed']} | primitive best "
                  f"changed_cell_acc={pb['changed_cell_acc']} exact={pb['exact_match']} (k_fit={pb['k_fit']}, "
                  f"movable={pb['movable']}) | llm_engine={le['changed_cell_acc'] if le else None} | "
                  f"primitive_beats_llm={r['primitive_beats_llm']}", flush=True)
            print(f"    learning curve: " + " ".join(
                f"k{c['k_fit']}={c['changed_cell_acc']}" for c in r["primitive_learning_curve"]), flush=True)
        else:
            print(f"\n  [{r['game']}] {r.get('verdict','no primitive fit')}", flush=True)
    fit_games = [r for r in rows if r.get("primitive_best") and r["primitive_best"]["exact_match"] > 0.5]
    best = max((r for r in rows if r.get("primitive_best")),
               key=lambda r: r["primitive_best"]["exact_match"], default=None)
    pair = {r["game"]: r for r in rows if r["game"] in ("ar25", "ka59")}
    verdict = ("complete_mechanic_primitive_translation_fits_" + "_".join(r["game"] for r in fit_games)
               if fit_games else
               "complete_mechanic_primitive_translation_fits_no_game_at_pixel_resolution_need_object_centric_repr")
    out = {"experiment": "arc3_mechanic_primitive_transfer", "games": GAMES,
           "similar_pair": ["ar25", "ka59"],
           "primitive_tested": "rigid_translation_of_a_selected_color (the simplest shared move mechanic)",
           "n_games_translation_primitive_fits": len(fit_games),
           "games_translation_primitive_fits": [r["game"] for r in fit_games],
           "best_fit_game": (best["game"] if best else None),
           "best_fit_exact_match": (best["primitive_best"]["exact_match"] if best else None),
           "similar_pair_share_low_level_dynamics": bool(
               "ar25" in pair and "ka59" in pair
               and pair["ar25"].get("primitive_best") and pair["ka59"].get("primitive_best")
               and pair["ar25"]["primitive_best"]["exact_match"] > 0.5
               and pair["ka59"]["primitive_best"]["exact_match"] > 0.5),
           "per_game": rows,
           "interpretation": (
               "A rigid-translation primitive (the simplest shared 'move a selected object' mechanic), "
               "params re-fit per game from few observations, was tested as the offline->live bridge. "
               "If exact_match is ~0 across games, the games' dynamics are NOT clean translations at the "
               "64x64 PIXEL representation (cell=1: logical-cell detection found no downsample) -- objects "
               "are multi-color sprites that reflect (ar25), push (ka59), or redraw, not rigid color "
               "translations. The bridge then needs (a) an OBJECT-CENTRIC representation (segment sprites, "
               "track them as objects) so primitives are clean, AND (b) a LIBRARY of primitives "
               "(translate/reflect/push/toggle), not one. This is the shared root cause across all three "
               "probes (seeding, whole-engine LOGO, this): the binding constraint is induction quality, "
               "which is gated by the state REPRESENTATION."),
           "honest_verdict": verdict,
           "inference_substrate": "offline_sim_no_quota_proposer_free_primitive_param_fit"}
    (REPO / "results" / "arc3_mechanic_primitive_transfer.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  translation primitive fits {len(fit_games)}/{len(GAMES)} games "
          f"(exact>0.5): {[r['game'] for r in fit_games]} | best={out['best_fit_game']}@{out['best_fit_exact_match']}", flush=True)
    print(f"  -> {verdict}\n  wrote results/arc3_mechanic_primitive_transfer.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
