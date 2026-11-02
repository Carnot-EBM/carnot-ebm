#!/usr/bin/env python3
"""tu93 LEVEL-2 hidden-state diagnostic — is L2's transition a deterministic function of the
VISIBLE logical grid, or does it carry HIDDEN STATE?

Mirrors scripts/experiments/experiment_8_hidden_state_diagnostic.py (which did this for
wa30/sb26/ka59 over L1 walks) but RE-ANCHORED to tu93's LEVEL 2: we must first SOLVE L1 to
arrive at L2, then collect (logical_grid, action, next_logical_grid) transitions IN L2 only.

Reaching L2 (per scripts/experiments/experiment_program_gen.py):
  offline_arcade().make(tu93) -> _warm(env, False) -> plan_in_model(L1) inside the hand-induced
  maze world model (results/arc_e3/tu93/world_model_nav.py) -> execute until _levels_completed==1.
  We VERIFY _levels_completed==1 before collecting anything.

Collecting genuine REPEATS in L2: tu93 is 4-direction maze nav (ACTION1=up,2=down,3=left,4=right)
and moves are reversible, so OSCILLATING (up<->down, left<->right) naturally revisits the same
logical avatar configuration via different histories. That is the precondition for the determinism
test: the SAME (grid_hash, action) key recurs, and we ask whether it always maps to the SAME next
grid. tu93 also decrements a move-counter strip each step; the LOGICAL grid (game-resolution avatar
maze) is what we hash, mirroring experiment_8's logical-grid branch (the branch that flagged wa30
0.53 hidden vs sb26 0.0 grid-deterministic).

Note: tu93 L2 dies DETERMINISTICALLY after a few steps on a naive walk (per
results/experiment_program_gen_tu93.json: death_step=3 over 4 trials). So the collector treats a
game-over as a RESET (re-solve L1, re-enter L2) rather than fabricating transitions past it, and
keeps walking until it has banked enough recurring keys.

nondeterminism_fraction = n_nondeterministic_pairs / n_repeated_pairs.
  0.0  -> L2 transition is fully determined by the visible logical grid (like sb26).
  >0   -> hidden state: same visible grid + same action -> different outcomes (like wa30 0.53).

OFFLINE, zero quota. verifier_is_oracle: false (this is a measurement, not a verifier claim)."""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed, _game_over
from carnot.agentic.arc_graph_explore import _warm
from carnot.agentic.arc_executable_world_model import detect_cell, to_logical, plan_in_model

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "experiment_tu93_l2_hidden_state.json"
GAME = "tu93"
MODEL_FILE = REPO / "results" / "arc_e3" / GAME / "world_model_nav.py"
NAV_ACTIONS = (1, 2, 3, 4)  # up, down, left, right
REVERSE = {1: 2, 2: 1, 3: 4, 4: 3}


def _ok(frame) -> bool:
    try:
        return np.asarray(grid_of(frame)).ndim == 2
    except Exception:
        return False


def _load_model():
    import importlib.util

    spec = importlib.util.spec_from_file_location("tu93_nav_model", MODEL_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.engine, mod.is_level_complete


def reach_l2(arc, engine, is_level_complete, cell):
    """Make tu93, warm, solve L1 by planning in the hand-induced model + executing, and return the
    env now sitting AT L2. Returns (env, frame, levels_completed) or (None, None, lvl) on failure.
    Mirrors experiment_program_gen.py's deepen loop but stops at the first level-up."""
    env = arc.make(GAME, scorecard_id=arc.open_scorecard())
    f = _warm(env, False)
    if not _ok(f):
        return None, None, -1
    start_level = _levels_completed(f)
    g = to_logical(grid_of(f), cell)
    plan = plan_in_model(engine, is_level_complete, g, max_nodes=40000, max_depth=80)
    if not plan:
        return None, None, start_level
    for step in plan:
        nf = env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))
        if nf is None or not _ok(nf):
            return None, None, _levels_completed(f) if _ok(f) else -1
        f = nf
        if _levels_completed(f) > start_level:
            break
    return env, f, _levels_completed(f)


def derive_counter_mask(arc, engine, is_level_complete, cell, pool):
    """Identify the per-step move-counter HUD cells from a DEDICATED THROWAWAY probe env (never a
    collection episode, so collection stays pristine). At the pristine L2 entry, take a BLOCKED move
    (one the grid-only engine predicts leaves the avatar in place): the cells that nonetheless change
    are exactly the ticking move-counter strip. If no blocked first move exists, returns an all-False
    mask (the full and masked branches then coincide, which is itself informative)."""
    env, f, lvl = reach_l2(arc, engine, is_level_complete, cell)
    if env is None or not _ok(f) or lvl < 1:
        return None
    g_before = np.asarray(to_logical(grid_of(f), cell))
    for a_try in pool:
        # fresh env per probe action so each probe sees the pristine L2 entry
        penv, pf, plvl = reach_l2(arc, engine, is_level_complete, cell)
        if penv is None or not _ok(pf) or plvl < 1:
            continue
        gb = np.asarray(to_logical(grid_of(pf), cell))
        pred = np.asarray(engine(gb.copy(), a_try, None))
        if not (pred.shape == gb.shape and np.array_equal(pred, gb)):
            continue  # engine says this move is NOT blocked; skip
        probe = penv.step(_game_action(GameAction, a_try), data=None)
        if probe is None or not _ok(probe) or _game_over(probe) or _levels_completed(probe) > plvl:
            continue
        ga = np.asarray(to_logical(grid_of(probe), cell))
        if np.array_equal(ga, gb):
            continue  # nothing changed at all (no visible counter); keep looking
        return ga != gb  # cells that changed on a blocked move == the move-counter HUD
    return np.zeros(g_before.shape, dtype=bool)


def collect_l2_transitions(arc, engine, is_level_complete, cell, rng, target_keys, max_resets,
                           steps_per_episode, n_action_pool):
    """Collect (visible_logical_grid_hash, action, next_visible_logical_grid_hash) transitions INSIDE
    L2. tu93's L2 entry state is byte-identical across fresh episodes (verified: hash 8e877eda every
    time) and the move-counter strip is episode-synchronized (it resets on each L2 entry). Therefore
    the repeat source that experiment_8 got from multi-rollout overlap we get from ACROSS-EPISODE
    PREFIX OVERLAP: each episode walks a short random sequence drawn from a small action pool, so the
    same (state, action) key recurs across episodes at matching walk depth, with the counter at the
    same value -> a genuine, fair repeat of the FULL visible grid.

    We also record a NAVIGATION-MASKED hash (the move-counter strip blanked) so we can locate WHERE
    any nondeterminism lives (experiment_8's raw-vs-logical split): if the full grid is
    nondeterministic but the masked grid is deterministic, the only 'hidden' state is the visible
    counter; if BOTH are deterministic the L2 transition is grid-determined; if the masked grid is
    nondeterministic there is genuine hidden state not in the visible grid.

    A game-over (tu93 L2 is fatal on some moves) ENDS the episode (we never record past death). We
    stop when enough recurring full-grid keys are banked or the reset budget is spent.

    Returns (records, counter_mask, n_resets, n_l2_episodes, l2_reached_ever).
    records: list of (full_hash, action, next_full_hash, masked_hash, next_masked_hash)."""
    records = []
    n_resets = 0
    n_l2_episodes = 0
    l2_reached_ever = False

    pool = list(NAV_ACTIONS[:n_action_pool]) if n_action_pool else list(NAV_ACTIONS)
    # Derive the move-counter HUD mask ONCE from a dedicated throwaway probe (keeps collection pristine).
    counter_mask = derive_counter_mask(arc, engine, is_level_complete, cell, pool)

    def _full_h(fr):
        return frame_hash(np.asarray(to_logical(grid_of(fr), cell)))

    def _masked_h(fr):
        g = np.asarray(to_logical(grid_of(fr), cell)).copy()
        if counter_mask is not None and counter_mask.any():
            g[counter_mask] = 0
        return frame_hash(g)

    def n_repeated_full_keys():
        counts = defaultdict(int)
        for fh, a, _, _, _ in records:
            counts[(fh, a)] += 1
        return sum(1 for c in counts.values() if c >= 2)

    while n_resets <= max_resets and n_repeated_full_keys() < target_keys:
        env, f, lvl = reach_l2(arc, engine, is_level_complete, cell)
        n_resets += 1
        if env is None or not _ok(f) or lvl < 1:
            continue  # failed to reach L2 this attempt; try again
        l2_reached_ever = True
        n_l2_episodes += 1

        # Short random walk from the (pristine) L2 entry; small pool -> prefixes collide across episodes.
        for _ in range(steps_per_episode):
            if not _ok(f):
                break
            a = pool[rng.randrange(len(pool))]
            fh, mh = _full_h(f), _masked_h(f)
            nf = env.step(_game_action(GameAction, a), data=None)
            if nf is None or not _ok(nf):
                break
            if _game_over(nf):
                break  # fatal move -> end episode; do NOT record past death
            if _levels_completed(nf) > lvl:
                break  # advanced beyond L2; stop (outside scope)
            records.append((fh, a, _full_h(nf), mh, _masked_h(nf)))
            f = nf
    return records, counter_mask, n_resets, n_l2_episodes, l2_reached_ever


def determinism(records, state_idx, next_idx):
    """Group by (grid_hash, action). Among groups that RECUR (seen >=2x), what fraction map to >1
    distinct next grid? That fraction is the hidden-state signal. state_idx/next_idx select the FULL
    (0,2) or MASKED (3,4) grid hash columns of a record tuple
    (full_h, action, next_full_h, masked_h, next_masked_h)."""
    counts = defaultdict(int)
    nexts = defaultdict(set)
    for r in records:
        key = (r[state_idx], r[1])
        counts[key] += 1
        nexts[key].add(r[next_idx])
    repeated = [(k, nexts[k]) for k, c in counts.items() if c >= 2]
    nondet = [k for k, ns in repeated if len(ns) >= 2]
    return {
        "n_distinct_keys": len(counts),
        "n_repeated_pairs": len(repeated),
        "n_nondeterministic_pairs": len(nondet),
        "nondeterminism_fraction": round(len(nondet) / len(repeated), 4) if repeated else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-keys", type=int, default=40, help="stop once this many recurring full-grid keys")
    ap.add_argument("--max-resets", type=int, default=400)
    ap.add_argument("--steps-per-episode", type=int, default=12,
                    help="short walk so prefixes collide across episodes (the repeat source)")
    ap.add_argument("--action-pool", type=int, default=3,
                    help="size of the nav-action pool (small -> more prefix collisions)")
    ap.add_argument("--seed", type=int, default=20260622)
    args = ap.parse_args()
    t0 = time.time()
    rng = random.Random(args.seed)

    engine, is_level_complete = _load_model()
    arc = kit.offline_arcade()

    # cell detection on a fresh warm frame
    f0 = _warm(arc.make(GAME, scorecard_id=arc.open_scorecard()), False)
    if not _ok(f0):
        artifact = {
            "experiment": "experiment_tu93_l2_hidden_state",
            "honest_verdict": "complete: blocked_degenerate_start_frame",
            "verifier_is_oracle": False,
            "inference_substrate": "offline_arc_search",
            "random_seed": args.seed,
            "l2_reachable": False,
            "duration_s": round(time.time() - t0, 1),
        }
        OUT.write_text(json.dumps(artifact, indent=2))
        print(json.dumps(artifact, indent=2))
        return 0
    cell = detect_cell(grid_of(f0))

    records, counter_mask, n_resets, n_l2_episodes, l2_reached_ever = collect_l2_transitions(
        arc, engine, is_level_complete, cell, rng,
        args.target_keys, args.max_resets, args.steps_per_episode, args.action_pool)

    # FULL visible logical grid (cols 0,2) is the primary test the task asks for; MASKED grid (cols
    # 3,4) blanks the per-step move-counter HUD to locate WHERE any nondeterminism lives.
    full = determinism(records, 0, 2)
    masked = determinism(records, 3, 4)
    repeated = full["n_repeated_pairs"]
    frac = full["nondeterminism_fraction"]
    mfrac = masked["nondeterminism_fraction"]

    if not l2_reached_ever:
        verdict = "complete: L2_unreachable_to_test_could_not_solve_L1_to_enter_L2"
    elif repeated == 0:
        verdict = "complete: indeterminate_zero_repeated_grid_action_keys_in_L2_cannot_test_determinism"
    elif frac is not None and frac == 0.0:
        verdict = ("complete: L2_transition_is_grid_deterministic_no_hidden_state_"
                   "same_visible_logical_grid_plus_action_always_same_next_grid")
    elif frac is not None and frac > 0.0 and (mfrac is not None and mfrac == 0.0):
        verdict = ("complete: L2_full_grid_nondeterministic_ONLY_via_visible_move_counter_"
                   "navigation_masked_grid_is_deterministic_no_truly_hidden_state")
    elif frac is not None and frac > 0.0:
        verdict = ("complete: L2_transition_has_HIDDEN_STATE_"
                   "same_visible_logical_grid_plus_action_yields_distinct_next_grids")
    else:
        verdict = "complete: indeterminate"

    artifact = {
        "experiment": "experiment_tu93_l2_hidden_state",
        "honest_verdict": verdict,
        "verifier_is_oracle": False,
        "inference_substrate": "offline_arc_search",
        "random_seed": args.seed,
        "game": GAME,
        "level_under_test": 2,
        "model_file": str(MODEL_FILE.relative_to(REPO)),
        "logical_cell_size": int(cell),
        "l2_reachable": bool(l2_reached_ever),
        "n_l2_episodes": int(n_l2_episodes),
        "n_resets_to_reach_l2": int(n_resets),
        "n_l2_transitions": len(records),
        "counter_mask_cells": int(counter_mask.sum()) if counter_mask is not None else 0,
        # PRIMARY: full visible logical grid
        "n_grid_action_pairs_with_repeats": int(repeated),
        "n_nondeterministic_pairs": int(full["n_nondeterministic_pairs"]),
        "nondeterminism_fraction": frac,
        "full_grid_determinism": full,
        # SECONDARY: navigation-masked grid (move-counter HUD blanked)
        "masked_grid_determinism": masked,
        "comparison_experiment_8": {
            "sb26_logical_nondeterminism_rate": 0.0,
            "wa30_logical_nondeterminism_rate": 0.53,
            "note": "0 == grid-deterministic (sb26 class); >0 == hidden state (wa30 class).",
        },
        "duration_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(artifact, indent=2))
    print(json.dumps(artifact, indent=2))
    print(f"\n  -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
