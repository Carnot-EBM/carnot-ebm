"""FULL ARC-AGI-3 PASS: run the generalized first-contact solver on EVERY offline game and produce an
accuracy scorecard + a failure-CLASS breakdown that names the new solver capabilities to research.

Instead of chasing individual games, this runs the whole pipeline (explore -> identify agent -> induce
goal -> plan -> solve, no banked solve) across all 25 offline games and classifies each outcome by the
STAGE that failed, so the research backlog is defined by failure CLASSES, not one-off games:

  SOLVED            -- first-contact L1 solve (real-env confirmed)
  FAIL_AGENT_ID     -- no agent identified (dynamics not translate/growth) -> richer object/dynamics model
  FAIL_EXPLORATION  -- agent found but win not stumbled               -> curiosity/directed exploration
  FAIL_GOAL_INDUCE  -- win stumbled but no goal object induced         -> better goal-object identification
  FAIL_SOLVE        -- goal induced but best_first_search found no win -> stronger heuristic / learned search

Measures L1 first-contact solvability (the current capability). Multi-level continuation is a noted
follow-on (induce the goal at each level, re-plan). Proposer-free, no banked solve, zero quota.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

ALL_GAMES = sorted(p.name for p in (REPO / "environment_files").iterdir() if p.is_dir())


def _pipeline():
    spec = importlib.util.spec_from_file_location(
        "ttgi", str(REPO / "scripts" / "experiments" / "arc3_test_time_goal_induction.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def classify(rec):
    s = rec.get("induced_goal_solve")
    if s and s.get("solved"):
        return "SOLVED", s.get("actions")
    if rec.get("agent") is None or (rec.get("agent") or {}).get("color") is None:
        return "FAIL_AGENT_ID", None
    if not rec.get("win_stumbled"):
        return "FAIL_EXPLORATION", None
    if rec.get("induced_goal_color") is None:
        return "FAIL_GOAL_INDUCE", None
    return "FAIL_SOLVE", None


RESEARCH = {
    "FAIL_AGENT_ID": "richer object/dynamics model -- non-translation mechanics (multi-colour sprites, "
                     "rotation, configuration/toggle agents) beyond translate+growth",
    "FAIL_EXPLORATION": "curiosity/directed exploration -- growth-aware + multi-instance state keys, "
                        "deeper/novelty-weighted coverage to stumble deep wins",
    "FAIL_GOAL_INDUCE": "better goal-object identification -- the win-relevant object/region when 'nearest "
                        "colour the agent reached' is ambiguous",
    "FAIL_SOLVE": "stronger planning -- learned/abstraction heuristics + subgoal decomposition for the "
                  "combinatorial tail (ka59-class); the open research problem",
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=",".join(ALL_GAMES))
    ap.add_argument("--explore-budget", type=int, default=5000)
    ap.add_argument("--max-exp", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    games = args.games.split(",")
    P = _pipeline()
    print(f"== FULL ARC-AGI-3 PASS: generalized first-contact solver on {len(games)} games ==", flush=True)
    rows, t0 = [], time.time()
    from collections import Counter
    classes = Counter()
    for g in games:
        gt = time.time()
        try:
            rec = P.run_game(g, 150, args.explore_budget, args.seed, args.max_exp)
        except Exception as ex:
            rec = {"game": g, "error": f"{type(ex).__name__}: {str(ex)[:120]}"}
        cls, actions = ("ERROR", None) if "error" in rec else classify(rec)
        classes[cls] += 1
        agent = (rec.get("agent") or {}).get("color")
        rows.append({"game": g, "class": cls, "actions": actions, "agent": agent,
                     "win_stumbled": rec.get("win_stumbled"), "goal": rec.get("induced_goal_color"),
                     "dur_s": round(time.time() - gt, 1)})
        tag = f"SOLVED@{actions}" if cls == "SOLVED" else cls
        print(f"  {g:5} {tag:22} agent={agent} stumbled={rec.get('win_stumbled')} [{rows[-1]['dur_s']:.0f}s]", flush=True)
    solved = [r["game"] for r in rows if r["class"] == "SOLVED"]
    n = len(rows)
    # research backlog: failure classes ordered by how many games they block
    backlog = [{"failure_class": c, "games_blocked": classes[c],
                "example_games": [r["game"] for r in rows if r["class"] == c][:6],
                "research_capability": RESEARCH.get(c, "n/a")}
               for c in sorted(classes, key=lambda c: -classes[c]) if c.startswith("FAIL")]
    out = {
        "experiment": "arc3_full_pass_scorecard",
        "n_games": n, "l1_first_contact_solved": len(solved), "solve_rate": round(len(solved) / max(1, n), 3),
        "solved_games": solved, "class_breakdown": dict(classes),
        "research_backlog_by_failure_class": backlog,
        "per_game": rows, "total_duration_s": round(time.time() - t0, 1),
        "scope": "L1 first-contact solvability; multi-level continuation is a noted follow-on",
        "honest_verdict": f"complete_full_pass_l1_solve_rate_{len(solved)}_of_{n}",
        "inference_substrate": "offline_arc_agi3_generalized_first_contact_no_banked_solve",
    }
    (REPO / "results" / "arc3_full_pass_scorecard.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  L1 FIRST-CONTACT SOLVE RATE: {len(solved)}/{n} ({out['solve_rate']:.0%})", flush=True)
    print(f"  class breakdown: {dict(classes)}", flush=True)
    print("  RESEARCH BACKLOG (failure class -> capability to add):", flush=True)
    for b in backlog:
        print(f"    [{b['games_blocked']:2} games] {b['failure_class']}: {b['research_capability'][:80]}", flush=True)
    print(f"  wrote results/arc3_full_pass_scorecard.json ({out['total_duration_s']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
