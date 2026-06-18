"""E3 induce->verify->plan PIPELINE DIAGNOSTIC -- find WHICH stage fails on the gap-1 games.

The stronger-proposer test (Qwen-35B vs gemma-12B) closed 0/6 with byte-identical action counts ->
the induced plan was a NO-OP regardless of model. So model size is not the lever; the PIPELINE is.
This instruments every stage on a gap-1 game so we fix the failing STAGE, not guess:

  1. EXPLORE      -- collect_transitions(n=80): how many transitions, and did explore even SEE a win
                     (level_after>level_before)? If 0 wins, the induced model has no positive example
                     and plan_in_model cannot target a win -- the explore BUDGET is the root cause.
  2. INDUCE       -- proposer.induce: did the LLM write a world_model.py at all?
  3. VERIFY       -- WorldModelVerifier.score: the induced engine's accuracy on the recorded
                     transitions. < 0.5 => the dynamics are wrong (induce quality is the bottleneck).
  4. WIN-PREDICATE-- test the induced is_level_complete on a TRUE win grid (from the banked solve). If
                     it does NOT fire on a real win, the planner can never find a win, whatever the engine.
  5. PLAN         -- plan_in_model: does BFS-in-the-model return a plan? (needs accurate engine + a
                     correct, reachable win predicate.)

Uses the gemma-12B proposer on ISOLATED port 8920 (model is incidental per the prior result; gemma is
faster). Zero quota. Tears down its llama-server on exit (the proposer leaks it otherwise).
"""
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
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_action

GAMES = ["cn04", "ar25"]          # gap-1 floor games (cn04 = the one v2 unlocked offline)
EXPLORE_N = 80


def _mh():
    spec = importlib.util.spec_from_file_location(
        "mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _true_win_grid(game: str, cell: int):
    """Replay the banked solution to the FIRST level-up; return the logical grid at the win (or None).
    Used to test whether the induced is_level_complete recognizes a REAL win."""
    mh = _mh()
    src = mh.RESOLVED_ARTIFACTS.get(game, mh.GAME_ARTIFACTS.get(game))
    acts = [a for a in (mh.load_actions(src) or []) if mh.normalize(a)[0] is not None]
    if not acts:
        return None
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env.reset()
    if game in mh.WARMUP_GAMES and acts:
        aid, data = mh.normalize(acts[0])
        f = env.step(getattr(GameAction, f"ACTION{aid}"), data=data)
    l0 = _levels_completed(f)
    for a in acts:
        aid, data = mh.normalize(a)
        f = env.step(getattr(GameAction, f"ACTION{aid}"), data=data)
        if f is None:
            return None
        if _levels_completed(f) > l0:
            return e3.to_logical(grid_of(f), cell)
    return None


def diagnose(game: str, proposer) -> dict:
    rec: dict = {"game": game}
    # 1. EXPLORE
    trans, cell = e3.collect_transitions(game, n=EXPLORE_N)
    n_wins = sum(1 for t in trans if t.level_after > t.level_before)
    rec["explore"] = {"n_transitions": len(trans), "cell": cell, "wins_seen_in_explore": n_wins}
    arc = kit.offline_arcade()
    root_grid = e3.to_logical(grid_of(arc.make(game, scorecard_id=arc.open_scorecard()).reset()), cell)
    # 2. INDUCE
    try:
        ok, msg = proposer.induce(game, trans, cell)
    except Exception as ex:
        ok, msg = False, f"{type(ex).__name__}: {str(ex)[:120]}"
    rec["induce"] = {"ok": bool(ok), "msg": str(msg)[:160]}
    if not ok:
        rec["verdict"] = "INDUCE_FAILED (proposer wrote no engine)"
        return rec
    # 3. LOAD + VERIFY
    try:
        engine, is_done = e3.load_engine(game)
    except Exception as ex:
        rec["verdict"] = f"LOAD_ENGINE_FAILED: {type(ex).__name__}: {str(ex)[:120]}"
        return rec
    vr = e3.WorldModelVerifier(trans).score(engine)
    rec["verify"] = {"accuracy": round(vr.accuracy, 3), "n_correct": vr.n_correct, "n": vr.n,
                     "sample_mismatch": vr.mismatches[:2]}
    # 4. WIN-PREDICATE sanity on a TRUE win grid
    wg = _true_win_grid(game, cell)
    if wg is not None and is_done is not None:
        try:
            rec["win_predicate"] = {"recognizes_true_win": bool(is_done(wg))}
        except Exception as ex:
            rec["win_predicate"] = {"recognizes_true_win": None, "error": str(ex)[:120]}
    else:
        rec["win_predicate"] = {"recognizes_true_win": None, "note": "no true-win grid or no is_done"}
    # 5. PLAN
    plan = e3.plan_in_model(engine, is_done, root_grid)
    rec["plan"] = {"returned": plan is not None, "len": len(plan) if plan else 0}
    # VERDICT -- which stage is the wall
    if n_wins == 0:
        rec["verdict"] = ("EXPLORE_SAW_NO_WIN: the 80-transition explore budget never reached a level-up, "
                          "so the induced model has no win example -> raise the explore budget / guide it.")
    elif vr.accuracy < 0.5:
        rec["verdict"] = (f"ENGINE_INACCURATE ({vr.accuracy:.0%} < 50%): the induced dynamics are wrong "
                          "-> better induce (more/cleaner transitions, prompt, or RFT), not a bigger model.")
    elif rec["win_predicate"].get("recognizes_true_win") is False:
        rec["verdict"] = ("WIN_PREDICATE_WRONG: the induced is_level_complete does not fire on a REAL win "
                          "-> the planner can never target a win, whatever the engine.")
    elif not rec["plan"]["returned"]:
        rec["verdict"] = ("PLAN_NOT_FOUND: engine accurate + win predicate ok, but BFS-in-model found no "
                          "win within budget -> the win is deep / model state-space too large; widen plan.")
    else:
        rec["verdict"] = "PLAN_FOUND (pipeline produced a plan -- check execution divergence next)."
    return rec


def main() -> int:
    print("== E3 pipeline diagnostic: WHICH stage fails on the gap-1 games? ==", flush=True)
    proposer = e3.LocalGGUFProposer(repo_substr="gemma-4-12B-it", port=8920)
    rows = []
    try:
        for g in GAMES:
            r = diagnose(g, proposer)
            rows.append(r)
            print(f"\n  [{g}] {json.dumps({k: r[k] for k in r if k != 'game'}, default=str)[:600]}", flush=True)
            print(f"  -> VERDICT: {r['verdict']}", flush=True)
    finally:
        # tear down the leaked llama-server (the proposer does not self-clean)
        try:
            import subprocess
            subprocess.run(["pkill", "-f", "llama-server.*8920"], timeout=10)
        except Exception:
            pass
    verdicts = [r.get("verdict", "").split(":")[0].split(" ")[0] for r in rows]
    out = {"experiment": "arc3_e3_pipeline_diagnostic", "games": GAMES, "per_game": rows,
           "failing_stages": verdicts,
           "honest_verdict": "complete_e3_pipeline_diagnostic_" + "_".join(verdicts).lower(),
           "inference_substrate": "offline_sim_no_quota_e3_local_gguf_induction_port8920"}
    (REPO / "results" / "arc3_e3_pipeline_diagnostic.json").write_text(json.dumps(out, indent=2))
    print(f"\n  failing stages: {verdicts}", flush=True)
    print(f"  wrote results/arc3_e3_pipeline_diagnostic.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
