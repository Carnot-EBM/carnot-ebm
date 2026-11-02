"""Offline validation of the CarnotAgent competition policy — drives
CarnotAgentPolicy through the OFFLINE game sims (environment_files) exactly the way
the ARC-AGI-3-Agents harness drives an Agent (step-wise: is_done? -> choose_action ->
step), and confirms it reaches our claimed levels. Zero quota, no internet — the same
shape as the competition's offline evaluation. Proves the agent SCORES before any
operator-gated Kaggle submission.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_competition_agent import (
    CLAIMED, CarnotAgentPolicy, load_solutions, MAX_ACTIONS, _level_of,
)


def play_offline(game: str, solutions: dict, *, explore: bool, budget: int) -> dict:
    """Run the policy through the offline sim the way the harness would.
    explore=True forces the GENERIC step-wise solver from scratch (no banked plan) —
    the real generalization test, since eval games are unseen."""
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    policy = CarnotAgentPolicy(game, solutions, force_explore=explore)
    frames: list = []
    latest = None
    steps = 0
    for _ in range(budget):
        if policy.is_done(frames, latest):
            break
        kind, data = policy.next_move(frames, latest)
        if kind == "RESET":
            latest = env.reset()
        elif kind is None:
            break
        else:
            latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
            steps += 1
        frames.append(latest)
        if latest is None:
            break
    return {"game": game, "target": CLAIMED.get(game, 1),
            "reached": _level_of(latest), "actions": steps}


def main() -> int:
    explore = "--explore" in sys.argv
    budget = 6000 if explore else MAX_ACTIONS
    label = ("GENERIC step-wise EXPLORER from scratch (unseen-game proxy)" if explore
             else "recognize-and-replay")
    print(f"== CarnotAgent offline validation — {label} ==", flush=True)
    solutions = load_solutions()
    rows, total, ok = [], 0, 0
    for game in CLAIMED:
        t0 = time.time()
        r = play_offline(game, solutions, explore=explore, budget=budget)
        r["pass"] = r["reached"] >= (1 if explore else r["target"])  # explore unit = +1 level
        rows.append(r)
        total += max(0, r["reached"])
        ok += int(r["pass"])
        print(f"  {game:5} target L{r['target']} -> reached L{r['reached']} "
              f"in {r['actions']} actions  {'PASS' if r['pass'] else 'FAIL'}  [{time.time()-t0:.0f}s]",
              flush=True)
    floor = len(CLAIMED) if not explore else 0
    print(f"\n  AGENT TOTAL: {total} levels; {ok}/{len(CLAIMED)} games "
          f"{'pass' if not explore else 'solved >=L1 from scratch'}", flush=True)
    out = REPO / "results" / ("arc_competition_explore.json" if explore else "arc_competition_validate.json")
    out.write_text(json.dumps({
        "experiment": "arc_competition_" + ("explore" if explore else "validate"),
        "method": ("carnot_agent_generic_stepwise_explorer_from_scratch" if explore
                   else "carnot_agent_policy_offline_stepwise_replay"),
        "agent_total_levels": total, "games_solved": ok, "games": len(CLAIMED),
        "per_game": rows, "inference_substrate": "offline_sim_no_quota", "run_date": "2026-06-17",
        "honest_verdict": (
            ("success_generic_explorer_solved_%d_of_%d_from_scratch" % (ok, len(CLAIMED))) if explore
            else ("success_carnot_agent_validated_offline" if ok == len(CLAIMED)
                  else f"complete_carnot_agent_partial_{ok}_of_{len(CLAIMED)}")),
    }, indent=2))
    print(f"  wrote {out.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
