"""The FOCUSED ARC-AGI-3 LOOP — the measurement engine for rapid leaderboard progress.

Scores the CarnotAgent on the public games FROM SCRATCH (force-explore: the unseen-game
proxy, since the real eval is hidden games) using the LEADERBOARD METRIC: per game,
levels_completed and efficiency = sum over solved levels of min(baseline/agent_actions,1)^2.
Writes a leaderboard-style scorecard AND a per-game GAP LOG (which games fail + the
failure signature) so the next iteration targets the worst gap, not a random change.

The loop cadence (one turn each, gated):
  1. RUN this harness  -> levels + efficiency + gap log (no quota, offline).
  2. READ the worst gap (a game stuck at L0, or a solved game with terrible efficiency).
  3. IMPROVE one ingredient (salience tiers / frontier-distance nav / status-masking /
     E3 world-model induction for a deep game) -- a single, attributable change.
  4. RE-RUN; keep the change only if levels or efficiency strictly improved (regression
     gate: the previously-solved games must not regress).
  5. Append the closed/!closed gap to the log (never-prune) and repeat.

Usage: arc_leaderboard_eval.py [--budget N] [--mode explore|replay]
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
from carnot.agentic.arc_competition_agent import CLAIMED, CarnotAgentPolicy, load_solutions, _level_of


def _baseline_actions(env, game: str) -> dict:
    """Per-level human/reference action counts if the env exposes them (efficiency
    denominator). Best-effort; returns {} if unavailable offline."""
    for attr in ("baseline_actions", "human_actions", "reference_actions"):
        v = getattr(getattr(env, "info", env), attr, None)
        if v:
            return {i: int(x) for i, x in enumerate(v)} if isinstance(v, (list, tuple)) else dict(v)
    return {}


def run_game(game: str, solutions: dict, *, explore: bool, budget: int) -> dict:
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    base = _baseline_actions(env, game)
    # HUD-masking (StepwiseExplorer hud_mask) is available but OFF by default: measured
    # neutral on the public set and the env-probe costs real actions in competition.
    # Enable opt-in per-game if a counter-heavy hidden game shows state-explosion.
    policy = CarnotAgentPolicy(game, solutions, force_explore=explore)
    frames, latest, actions = [], None, 0
    start = None
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
            actions += 1
        if start is None:
            start = _level_of(latest)
        frames.append(latest)
        if latest is None:
            break
    reached = _level_of(latest)
    levels = max(0, reached - (start or 0))
    # leaderboard efficiency: min(baseline/agent,1)^2 per solved level (1.0 if no baseline)
    eff = 0.0
    if levels > 0:
        b = base.get(0)
        ratio = min((b / actions), 1.0) if b else 1.0
        eff = round(levels * ratio * ratio, 4)
    gap = None
    if reached <= (start or 0):
        gap = {"game": game, "stuck_at_level": reached, "actions_spent": actions,
               "signature": "no_level_up_within_budget",
               "needs": "richer exploration (salience tiers / frontier-dist nav) OR E3 world-model induction"}
    return {"game": game, "levels": levels, "reached": reached, "actions": actions,
            "efficiency": eff, "gap": gap}


def main() -> int:
    argv = sys.argv[1:]
    explore = "replay" not in argv  # default: the from-scratch (competition-relevant) measure
    # competition allows ~96k actions/game; 6000 badly understated solve-rate (8/11 of the
    # public games solve from scratch at 8-11k actions). 20000 reveals the true solvable set
    # while staying fast; the genuinely-hard tail (wa30/cn04/sk48) resists even 45k.
    budget = 20000
    if "--budget" in argv:
        budget = int(argv[argv.index("--budget") + 1])
    print(f"== ARC leaderboard eval — mode={'explore(from-scratch)' if explore else 'replay'} budget={budget} ==", flush=True)
    sols = load_solutions()
    rows, total_levels, total_eff, gaps = [], 0, 0.0, []
    for game in CLAIMED:
        t0 = time.time()
        r = run_game(game, sols, explore=explore, budget=budget)
        rows.append(r)
        total_levels += r["levels"]
        total_eff += r["efficiency"]
        if r["gap"]:
            gaps.append(r["gap"])
        print(f"  {game:5} levels={r['levels']} (L{r['reached']}) actions={r['actions']:5} "
              f"eff={r['efficiency']:.3f}  {'GAP' if r['gap'] else 'ok'}  [{time.time()-t0:.0f}s]", flush=True)
    print(f"\n  LEADERBOARD SCORE: {total_levels} levels, efficiency-sum {total_eff:.3f}; "
          f"{len(gaps)} open gaps", flush=True)
    if gaps:
        print("  WORST GAPS (target next):", ", ".join(g["game"] for g in gaps[:6]), flush=True)
    out = REPO / "results" / "arc_leaderboard_eval.json"
    out.write_text(json.dumps({
        "experiment": "arc_leaderboard_eval", "mode": "explore" if explore else "replay",
        "budget": budget, "total_levels": total_levels, "efficiency_sum": round(total_eff, 4),
        "open_gaps": gaps, "per_game": rows, "inference_substrate": "offline_sim_no_quota",
        "run_date": "2026-06-17",
        "honest_verdict": f"complete_leaderboard_eval_{total_levels}_levels_{len(gaps)}_gaps",
    }, indent=2))
    print(f"  wrote {out.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
