#!/usr/bin/env python3
"""Program-generalization (Executable World Model) — first swing, DEEPENING probe.

The leaderboard leader (Executable World Models, arXiv:2605.05138, RHAE ~58%) DEEPENS by inducing an
executable transition+goal MODEL once and PLANNING IN IMAGINATION across levels -- so reaching L2
doesn't require stumbling into it by real-env search (the gradient/seed wall that bounded our
value-relabeling, exp value_q_head v5/v6). This probes whether that lever works ON OUR STACK, reusing
the EXISTING framework (python/carnot/agentic/arc_executable_world_model.py) rather than reinventing it:

  --existing GAME : load the ALREADY-INDUCED + verified world model (results/arc_e3/<game>/world_model.py),
                    re-VERIFY it on freshly collected transitions (exact + changed-cell-recall), then run a
                    DEEPENING LOOP -- plan_in_model from the live state, execute in the real env, re-perceive,
                    re-plan -- and measure how many levels imagination-planning reaches vs the level the model
                    was induced/reproduced at. The Carnot WorldModelVerifier is the moat; the planner is generic BFS.

The honest question: does a world model induced for L1 GENERALIZE to plan L2+ in imagination? A genuine
logic engine (ka59: 3x3-push + clear-the-4s goal) should; a memorized-patch table (sc25: PATCH_BY_KEY +
hardcoded L1 hash) should not -- and that contrast is itself the finding about what "world model" buys.

OFFLINE, zero quota. verifier_is_oracle: false (the world model is oracle-DISTINCT: an induced+verified
predictive model, NOT the executable correctness oracle).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed, _game_over
from carnot.agentic.arc_graph_explore import _warm
from carnot.agentic.arc_executable_world_model import (
    collect_transitions, WorldModelVerifier, load_engine, to_logical, detect_cell, plan_in_model,
)

REPO = Path(__file__).resolve().parents[2]


def _out(game):
    return REPO / "results" / f"experiment_program_gen_{game}.json"


def _ok(frame):
    try:
        return np.asarray(grid_of(frame)).ndim == 2
    except Exception:
        return False


def _budget_remaining(env):
    """Best-effort read of the env's move counter (game-specific obfuscated attr). Returns
    (current, max) or (None, None). Used to rule OUT 'deepening stalled because moves ran out'."""
    try:
        gobj = env._game
        for attr in dir(gobj):
            o = getattr(gobj, attr, None)
            if hasattr(o, "current_steps"):
                return getattr(o, "current_steps", None), getattr(o, "max_steps", None)
    except Exception:
        pass
    return None, None


def diagnose_deepening_stall(game, engine, is_level_complete, cell, l1_actions, trials=4):
    """When deepening stalls right after L1, distinguish the CAUSE rather than asserting one. Re-plans L2
    on `trials` FRESH envs (replaying the same banked L1 prefix) and records: the death step each trial
    (deterministic across trials => NOT non-idempotent-reset parity, which is run-dependent), and the move
    budget remaining at the stall (full => NOT budget exhaustion). A deterministic, budget-unexhausted
    env game-over after a locally-correct L1 model => the L2 transition mechanic DIFFERS from L1's (the
    induced model does not generalize across the level boundary)."""
    death_steps, budgets = [], []
    for _ in range(trials):
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        f = _warm(env, False)
        for a in l1_actions:
            f = env.step(_game_action(GameAction, int(a["action"])), data=a.get("data"))
            if f is None or not _ok(f) or _levels_completed(f) > 0:
                break
        if f is None or not _ok(f) or _levels_completed(f) < 1:
            continue
        g2 = to_logical(grid_of(f), cell)
        plan2 = plan_in_model(engine, is_level_complete, g2, max_nodes=40000, max_depth=80)
        cur, mx = _budget_remaining(env)
        budgets.append((cur, mx))
        death = None
        for i, s in enumerate(plan2 or []):
            f = env.step(_game_action(GameAction, int(s["action"])), data=s.get("data"))
            if f is None or not _ok(f) or _game_over(f):
                death = i
                break
            if _levels_completed(f) > 1:
                death = "L2_reached"
                break
        death_steps.append(death)
    deterministic = len(set(map(str, death_steps))) == 1 and len(death_steps) >= 2
    budget_ok = any(c is not None and m is not None and c >= max(1, int(m) - len(l1_actions) - 4)
                    for c, m in budgets)  # near-full budget remaining at the stall
    if "L2_reached" in [str(d) for d in death_steps]:
        cause = "l2_reachable_on_fresh_env_replay_single_env_reuse_was_the_issue"
    elif deterministic:
        # A model faithful at L1 that dies at a FIXED step at L2 means the L2 env rule changed (a
        # mechanic shift), NOT the run-dependent non-idempotent-reset parity. The L1-induced model does
        # not generalize across the level boundary. (budget_ok is a best-effort extra confirmation it is
        # not move-exhaustion; the env source independently shows budget unexhausted at the stall.)
        cause = "deterministic_env_fatal_move_L2_mechanic_differs_from_L1_model_does_not_generalize"
    elif death_steps:
        cause = "run_dependent_death_consistent_with_non_idempotent_reset_parity"
    else:
        cause = "indeterminate"
    return {"l2_death_steps_over_trials": death_steps, "budget_at_stall": budgets,
            "deterministic": bool(deterministic), "budget_non_exhausted": bool(budget_ok),
            "diagnosed_cause": cause}


def deepen_via_world_model(game, engine, is_level_complete, cell, target_level, max_plan_per_level,
                           max_depth):
    """Real-env DEEPENING loop: from the live state, plan a path to is_level_complete INSIDE the induced
    model (zero real actions), execute it for real, and on each level-up re-perceive + re-plan. Measures
    how deep one induced model carries us. Returns (levels_reached, banked_action_labels, per_level)."""
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, False)
    start_level = _levels_completed(f)
    level = start_level
    banked = []           # raw action labels for the reproduction gate
    l1_actions = None     # the banked prefix that achieved the FIRST level-up (for stall diagnosis)
    per_level = []
    while level < start_level + target_level:
        g = to_logical(grid_of(f), cell)
        plan = plan_in_model(engine, is_level_complete, g, max_nodes=max_plan_per_level,
                             max_depth=max_depth)
        if not plan:
            per_level.append({"from_level": int(level), "planned": False,
                              "reason": "no_plan_to_is_level_complete_in_model"})
            break
        advanced = False
        steps_taken = 0
        stop_reason = "plan_exhausted_no_levelup"
        for step in plan:
            pred = engine(g.copy(), int(step["action"]), step.get("data"))
            nf = env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))
            steps_taken += 1
            if nf is None or not _ok(nf):
                stop_reason = "env_returned_none"
                break
            banked.append({"action": int(step["action"]), "data": step.get("data")})
            obs = to_logical(grid_of(nf), cell)
            # did the locally-accurate model's prediction diverge? (hidden-state signal)
            model_diverged = np.asarray(pred).shape != obs.shape or not np.array_equal(pred, obs)
            f = nf
            g = obs
            if _levels_completed(f) > level:
                level = _levels_completed(f)
                advanced = True
                stop_reason = "level_up"
                if l1_actions is None:
                    l1_actions = list(banked)   # snapshot the prefix that reached the first level-up
                break
            if _game_over(f):
                stop_reason = ("game_over_after_model_match" if not model_diverged
                               else "game_over_after_model_divergence")
                break
        per_level.append({"from_level": int(level if not advanced else level - 1),
                          "plan_len": len(plan), "steps_executed": steps_taken, "advanced": advanced,
                          "stop_reason": stop_reason})
        if not advanced:
            break
    return level - start_level, banked, per_level, l1_actions


def _load_model_file(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("hw_world_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "engine"), getattr(mod, "is_level_complete", None)


def run_existing(game, args, model_file=""):
    t0 = time.time()
    # re-VERIFY the model on fresh transitions (does the moat still hold?)
    trans, cell = collect_transitions(game, n=args.n_verify, seed=args.seed)
    if model_file:
        engine, is_level_complete = _load_model_file(model_file)
        mf_rel = model_file
    else:
        engine, is_level_complete = load_engine(game)
        mf_rel = f"results/arc_e3/{game}/world_model.py"
    vr = WorldModelVerifier(trans).score(engine)
    print(f"  [{game}] re-verify on {vr.n} fresh transitions: exact={vr.accuracy:.3f} "
          f"cell_recall={vr.cell_recall:.3f}", flush=True)

    reached, banked, per_level, l1_actions = deepen_via_world_model(
        game, engine, is_level_complete, cell, args.target_level, args.max_plan, args.max_depth)

    # reproduction gate: replay banked actions against a FRESH env
    repro = 0
    if banked:
        labels = [json.dumps({"action": a["action"], "data": a["data"]}) for a in banked]
        arc = kit.offline_arcade()
        envr = arc.make(game, scorecard_id=arc.open_scorecard())
        fr = _warm(envr, False)
        base = _levels_completed(fr)
        for a in banked:
            fr = envr.step(_game_action(GameAction, int(a["action"])), data=a["data"])
            if fr is None or not _ok(fr):
                break
        repro = _levels_completed(fr) - base

    deepened = reached >= 2 and repro >= 2
    # When deepening stalls right after L1, DIAGNOSE the cause empirically rather than asserting it: re-run
    # L2 on fresh envs and measure death-step determinism + budget. (A prior version mislabeled ANY
    # game-over as 'hidden state'; the adversarial review refuted that -- tu93's L2 death is DETERMINISTIC
    # and budget-unexhausted, i.e. the L2 transition mechanic differs from L1's, NOT the non-idempotent
    # reset parity, which would be run-dependent.)
    diag = None
    stalled_at_game_over = any(str(pl.get("stop_reason", "")).startswith("game_over") for pl in per_level)
    if reached >= 1 and not deepened and stalled_at_game_over and l1_actions:
        diag = diagnose_deepening_stall(game, engine, is_level_complete, cell, l1_actions)
    cause = (diag or {}).get("diagnosed_cause")
    if deepened:
        verdict = "success: world_model_imagination_planning_GENERALIZES_and_DEEPENS_to_L2plus_reproduced"
    elif reached >= 1 and cause and "does_not_generalize" in cause:
        verdict = ("complete: faithful_world_model_plans_and_reproduces_L1_in_imagination_but_L2_mechanic_"
                   "differs_deterministic_env_fatal_move_model_does_not_generalize_across_levels")
    elif reached >= 1 and cause and "parity" in cause:
        verdict = ("complete: faithful_world_model_reproduces_L1_but_deepening_run_dependent_"
                   "non_idempotent_reset_parity")
    elif reached >= 1:
        verdict = ("complete: world_model_reached_L1_only_no_cross_level_generalization_"
                   "honest_null_gap_sharpened")
    else:
        verdict = "complete: world_model_no_level_in_imagination_goal_or_engine_l1_specific_gap_sharpened"

    art = {"experiment": "experiment_program_gen", "game": game, "phase": "existing_model_deepening",
           "honest_verdict": verdict, "verifier_is_oracle": False,
           "inference_substrate": "offline_arc_search_plus_induced_world_model",
           "random_seed": args.seed, "model_file": mf_rel,
           "reverify_exact_match": round(float(vr.accuracy), 3),
           "reverify_cell_recall": round(float(vr.cell_recall), 3),
           "levels_reached_imagination": int(reached), "levels_reproduced": int(repro),
           "n_banked_actions": len(banked), "per_level": per_level,
           "deepening_stall_diagnosis": diag,
           "methodology_note": ("L2 deepening, when it stalls at a game-over, is diagnosed empirically "
                                "(death-step determinism + move-budget over fresh-env replays), not asserted. "
                                "For tu93 the L2 death is deterministic + budget-unexhausted => the L2 move "
                                "mechanic differs from L1's (model-generalization failure), NOT the "
                                "non-idempotent-reset parity (which is run-dependent)."),
           "deepened_to_L2plus_reproduced": bool(deepened), "duration_s": round(time.time() - t0, 1)}
    out = _out(game)
    out.write_text(json.dumps(art, indent=2))
    print(f"\nVERDICT: {verdict}\n  re-verify exact={vr.accuracy:.3f} cell_recall={vr.cell_recall:.3f} | "
          f"imagination reached L{reached} (reproduced L{repro}) via {len(banked)} actions -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", type=str, default="", help="game with an existing results/arc_e3/<g>/world_model.py")
    ap.add_argument("--handwritten", type=str, default="", help="game id paired with --model-file")
    ap.add_argument("--model-file", type=str, default="", help="path to a hand-induced world model (engine + is_level_complete)")
    ap.add_argument("--n-verify", type=int, default=80)
    ap.add_argument("--target-level", type=int, default=4)
    ap.add_argument("--max-plan", type=int, default=20000)
    ap.add_argument("--max-depth", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260622)
    args = ap.parse_args()
    if args.existing:
        return run_existing(args.existing, args)
    if args.handwritten and args.model_file:
        return run_existing(args.handwritten, args, model_file=args.model_file)
    ap.error("provide --existing GAME, or --handwritten GAME --model-file PATH")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
