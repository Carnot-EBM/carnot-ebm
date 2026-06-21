"""Local ARC-AGI-3 competition DEV-HARNESS: run the full agent stack against the offline API simulator
under a GLOBAL rate-limited step budget, with per-game give-up + bank-saved-budget + full instrumentation.

Operator directive (2026-06-21): the RATE LIMIT is the real bottleneck -- 10 real steps/sec over the eval
wall (preview cap 8h => ~288k steps; the 12h is the Kaggle NOTEBOOK cap incl. model-load overhead, 2026
play-cap UNCONFIRMED) across ~110 HIDDEN games. So we must (a) run the ENTIRE budget making useful use of
every API call, (b) allocate it across games, (c) give up on a game when its slice is spent and move on,
(d) BANK saved budget from a game won/abandoned early for later games, (e) instrument every component so
the outer loop can refine each toward the highest score. This harness is that substrate -- it reuses the
deterministic offline arcade (arc_solver_kit.offline_arcade, a LOCAL API simulator over environment_files)
+ the AUTHORITATIVE scorer (arc_agi.scorecard.EnvironmentScoreCalculator, via arc_leaderboard_eval.run_game)
so the local score cannot drift from the leaderboard.

The local sim has only the 25 PUBLIC games; they are the proxy for the hidden eval (the only offline
signal). Dev loop: RUN -> read the per-game gap + budget-utilization instrumentation -> refine ONE
component (the learned-dynamics engine, the give-up watchdog, the pacer, the meta-strategy) -> RE-RUN ->
keep the change only if total score strictly improved. This is the harness the operator's stack iterates in.

Build status: v1 = global step-budget allocation + bank-saved-budget + per-game cap/give-up (via the cap +
policy.is_done) + instrumentation + total-score readout, on the existing policy. NEXT refinements (each
validated by re-running this harness): the LiveTTTWorldModel learned engine (arc_live_ttt), the COLD
no-progress give-up watchdog, a step-pacing controller, cross-attempt memory weighting, the multi-engine
parallel composition, and the meta-strategy layer.

Offline, zero quota. CPU for the explorer policy (fast dev loop); the e3 policy needs the local GGUF.

Usage:
  .venv/bin/python scripts/arc_compete_sim.py [--policy explorer|e3] [--total-budget 288000]
      [--games g1,g2,...] [--min-cap 200] [--max-cap 8000]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))


def _arg(argv: list[str], flag: str, default: str) -> str:
    return argv[argv.index(flag) + 1] if flag in argv else default


# 8h preview wall x 10 steps/sec. The pacer is PARAMETERIZED; this is the default, not a hard assumption.
DEFAULT_TOTAL_BUDGET = 288_000


def run(games: list[str], policy_kind: str, total_budget: int, min_cap: int, max_cap: int) -> dict:
    # import here so the module loads even where arc_leaderboard_eval's heavy deps are absent
    import importlib.util

    spec = importlib.util.spec_from_file_location("lbe", str(REPO / "scripts" / "arc_leaderboard_eval.py"))
    lbe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lbe)  # type: ignore

    remaining = total_budget
    rows: list[dict] = []
    print(f"== ARC COMPETE-SIM dev-harness — policy={policy_kind} total_budget={total_budget} "
          f"games={len(games)} (offline sim, authoritative scorer) ==", flush=True)
    for i, game in enumerate(games):
        remaining_games = len(games) - i
        # DYNAMIC per-game cap: remaining budget evenly over remaining games, clamped. A game won/abandoned
        # early uses < cap, so `remaining` stays high -> later games get a bigger slice (bank-saved-budget).
        cap = max(min_cap, min(max_cap, remaining // max(1, remaining_games)))
        t0 = time.time()
        try:
            r = lbe.run_game(game, lbe._build_policy(policy_kind, game), budget=cap)
        except Exception as e:
            rows.append({"game": game, "error": f"{type(e).__name__}: {e}"[:160], "allocated_cap": cap})
            print(f"  {game:5} ERROR {type(e).__name__}", flush=True)
            continue
        used = int(r.get("actions", 0))
        remaining -= used
        gave_up = r["reached"] <= 0 and r["levels"] == 0
        rows.append({
            "game": game, "allocated_cap": cap, "actions_used": used,
            "banked_saved": max(0, cap - used), "levels": r["levels"], "reached": r["reached"],
            "efficiency": r["efficiency"], "gave_up_no_progress": gave_up,
            "actions_to_first_levelup": r.get("actions_to_first_levelup"),
            "budget_remaining_after": remaining,
        })
        print(f"  {game:5} cap={cap:5} used={used:5} saved={max(0, cap-used):5} L+{r['levels']} "
              f"eff={r['efficiency']:.4f} {'GAVE-UP' if gave_up else ''} [{time.time()-t0:.0f}s] "
              f"remaining={remaining}", flush=True)

    scored = [r for r in rows if "efficiency" in r]
    total_eff = round(sum(r["efficiency"] for r in scored), 4)
    total_levels = sum(r["levels"] for r in scored)
    total_used = sum(r["actions_used"] for r in scored)
    reached_any = [r for r in scored if r["levels"] > 0]
    out = {
        "experiment": "arc_compete_sim",
        "honest_verdict": f"complete_compete_sim_{policy_kind}_score_eff_{total_eff}_levels_{total_levels}",
        "policy": policy_kind, "total_budget": total_budget,
        "leaderboard_efficiency_sum": total_eff, "total_levels": total_levels,
        "games_reached_progress": len(reached_any), "games_total": len(scored),
        "total_steps_used": total_used,
        "budget_utilization": round(total_used / max(1, total_budget), 4),
        "budget_unused_banked": max(0, total_budget - total_used),
        "per_game": rows,
        "inference_substrate": ("verifier_ensemble_against_cached_candidates -- offline arcade multi-game "
                                "scored run under a global step budget; no live LLM unless policy=e3"),
        "verifier_is_oracle": False,
        "dev_loop_note": ("RUN -> read worst per-game gap + budget_utilization -> refine ONE component -> "
                          "RE-RUN -> keep only if leaderboard_efficiency_sum strictly improves."),
    }
    (REPO / "results" / "arc_compete_sim.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  SCORE: efficiency-sum {total_eff}, {total_levels} levels across "
          f"{len(reached_any)}/{len(scored)} games; budget used {total_used}/{total_budget} "
          f"({out['budget_utilization']:.1%}), banked {out['budget_unused_banked']}. "
          f"-> {out['honest_verdict']}", flush=True)
    return out


def main(argv: list[str]) -> int:
    policy_kind = _arg(argv, "--policy", "explorer")
    total_budget = int(_arg(argv, "--total-budget", str(DEFAULT_TOTAL_BUDGET)))
    min_cap = int(_arg(argv, "--min-cap", "200"))
    max_cap = int(_arg(argv, "--max-cap", "8000"))
    games_arg = _arg(argv, "--games", "")
    if games_arg:
        games = games_arg.split(",")
    else:  # default to the public games we have offline sims + baselines for
        import importlib.util

        spec = importlib.util.spec_from_file_location("lbe", str(REPO / "scripts" / "arc_leaderboard_eval.py"))
        lbe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lbe)  # type: ignore
        games = list(lbe.CLAIMED)
    run(games, policy_kind, total_budget, min_cap, max_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
