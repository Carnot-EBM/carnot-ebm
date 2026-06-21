"""Solve test: does a plan BFS'd through the cell-recall-gated TTT engine actually SOLVE (reach a real
level-up in the offline sim) -- and is it GENERALIZATION or just memorized replay? (2026-06-21)

The cell-recall gate FIRES the prior+TTT path on 4/5 unseen games (arc_ttt_loo_gate_probe). But firing is
necessary-not-sufficient. plan_and_execute halts on EXACT prediction!=observation, and the engine checks the
L0 MEMORIZED table before the CNN. So two distinct things can produce a "solve":
  FULL (L0 + CNN): can solve by REPLAYING a memorized winning path -- NOT generalization (the registry's
                   existing 33-level replay already does this).
  CNN-ONLY (L0 disabled): must navigate to a win via the LEARNED CNN dynamics -- the true generalization
                   test. A cell-recall (non-exact) model is expected to DIVERGE on execution and halt.
The contrast isolates whether the prior+TTT adds NEW solving power vs just re-deriving the known replay.
Offline sim, zero-quota.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import json

from carnot.agentic.arc_transition_capture import TransitionCorpus
from carnot.agentic.arc_live_ttt import LiveTTTWorldModel, _load_prior, action_key
from carnot.agentic import arc_executable_world_model as e3

FIRING = ["ka59", "sc25", "tn36", "lp85"]  # the 4 games the cell-recall gate fired on
MAX_PLAN = 1500
MAX_DEPTH = 25


def make_cnn_only(ttt):
    """Engine that SKIPS the L0 memorized table -- forces the learned CNN dynamics (pure generalization)."""
    def eng(g, a, d):
        return ttt._l1.predict(g, action_key(a, d)) if ttt._l1 is not None else g
    return eng


def main() -> int:
    corpus = TransitionCorpus()
    prior = _load_prior("models/arc_dynamics_prior.pt")
    rows = []
    full_solved = cnn_solved = 0
    print("=== TTT solve test on the 4 cell-recall-firing games: FULL(L0+CNN) vs CNN-ONLY(generalization) ===",
          flush=True)
    print(f"{'game':6} {'wins':>4} | {'FULL plan':>9} {'FULL lvlup':>10} | {'CNN plan':>8} {'CNN lvlup':>9} "
          f"{'CNN reason':>28}", flush=True)
    for game in FIRING:
        tr = corpus.load(game)
        ttt = LiveTTTWorldModel(game, dynamics_backend="cnn", prior_state=prior)
        for t in tr:
            ttt.observe_transition(t)
        ttt.fit_now()
        nwin = len(ttt._win_states)
        try:
            full = e3.plan_and_execute(game, ttt.engine, ttt.is_level_complete,
                                       max_plan=MAX_PLAN, max_depth=MAX_DEPTH)
        except Exception as exc:
            full = {"error": f"{type(exc).__name__}: {exc}"[:80]}
        try:
            gen = e3.plan_and_execute(game, make_cnn_only(ttt), ttt.is_level_complete,
                                      max_plan=MAX_PLAN, max_depth=MAX_DEPTH)
        except Exception as exc:
            gen = {"error": f"{type(exc).__name__}: {exc}"[:80]}
        full_solved += int(bool(full.get("level_up")))
        cnn_solved += int(bool(gen.get("level_up")))
        rows.append({"game": game, "n_win_states": nwin, "full": full, "cnn_only": gen})
        print(f"{game:6} {nwin:>4} | {str(full.get('planned')):>9} {str(full.get('level_up')):>10} | "
              f"{str(gen.get('planned')):>8} {str(gen.get('level_up')):>9} "
              f"{str(gen.get('reason') or gen.get('error'))[:28]:>28}", flush=True)
    print(f"\nSOLVED (real level-up): FULL(L0+replay) {full_solved}/4   CNN-ONLY(generalization) {cnn_solved}/4",
          flush=True)
    out = {
        "experiment": "arc_ttt_solve_test",
        "honest_verdict": f"complete_ttt_solve_full_{full_solved}_cnnonly_{cnn_solved}_of_4",
        "full_solved_via_L0_or_cnn": full_solved,
        "cnn_only_generalization_solved": cnn_solved,
        "max_plan": MAX_PLAN, "max_depth": MAX_DEPTH, "per_game": rows,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": False,
        "interpretation": ("CNN-ONLY level-ups = NEW generalization solving power from the prior+TTT (beyond "
                           "the registry's memorized replay). FULL-but-not-CNN solves are L0 replay, which the "
                           "registry already banks. If CNN-ONLY solves ~0, the cell-recall model fires the gate "
                           "but cannot drive the exact-match plan_and_execute loop -- it improves prediction, "
                           "not solving, and won't move the 0.08 hidden score without a divergence-tolerant "
                           "executor (replan-on-divergence) + goal-induction."),
    }
    (REPO / "results" / "arc_ttt_solve_test.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"-> {out['honest_verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
