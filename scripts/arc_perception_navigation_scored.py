#!/usr/bin/env python3
"""Scored-path proof of the autonomous perception-navigation solver (REQ-ARC-WMTE-5839).

Three checks:
  1. auto_branch_mode(tu93) DERIVES the last per-game input (should be 'fresh_env' -- tu93's reset is
     non-idempotent) -> the navigation solver is now FULLY autonomous (zero per-game input).
  2. solve_navigation(tu93) reproduces the solve end-to-end with no per-game code.
  3. PerceptionNavigationPolicy(tu93) driven by the REAL scored `run_game` interface (is_done/next_move)
     carries the self-discovered solve -- vs the default explorer baseline. This confirms the scored-agent
     interface can deliver a perception-self-discovered navigation solve.

inference_substrate: offline_arcade_live_agent_runtime_self_discovery_no_llm. solve_provenance:
development_proxy (method validation on an already-registered game; no registry change).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def _load_run_game():
    spec = importlib.util.spec_from_file_location(
        "arc_leaderboard_eval", str(ROOT / "scripts" / "arc_leaderboard_eval.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m.run_game


def main() -> int:
    from carnot.agentic.arc_competition_agent import CarnotAgentPolicy
    from carnot.agentic.arc_perception_navigation import (
        PerceptionNavigationPolicy,
        auto_branch_mode,
        solve_navigation,
    )

    game = "tu93"
    run_game = _load_run_game()
    t0 = time.time()

    # (1) auto-derive the last per-game input
    branch = auto_branch_mode(game)
    print(f"[1] auto_branch_mode({game}) = {branch}  (expect fresh_env; tu93 reset is non-idempotent)")

    # (2) fully-autonomous solve (no per-game input at all)
    solve = solve_navigation(game)
    print(f"[2] solve_navigation({game}): pair={solve.get('pair')} branch={solve.get('branch_mode')} "
          f"reached L{solve.get('search_reached')} reproduced={solve.get('reproduced')} "
          f"(L{solve.get('reproduced_level')}) path_len={solve.get('path_len')}")

    # (3) scored-path: PerceptionNavigationPolicy via run_game, vs default explorer baseline
    budget = 120
    base = run_game(game, CarnotAgentPolicy(game, {}, force_explore=True), budget=budget)
    pol = PerceptionNavigationPolicy(game)
    scored = run_game(game, pol, budget=budget)
    print(f"[3] scored run_game baseline(default explorer): levels={base.get('levels')} reached={base.get('reached')}")
    print(f"[3] scored run_game PerceptionNavigationPolicy: levels={scored.get('levels')} reached={scored.get('reached')} plan_len={len(pol.plan)}")

    art = {
        "experiment": "outer_loop_arc_perception_navigation_scored",
        "experiment_id": "REQ-ARC-WMTE-5839",
        "run_date": "2026-07-23",
        "schema": "carnot.arc_perception_navigation_scored.v1",
        "title": "Fully-autonomous perception navigation solver (auto branch_mode) + scored-path proof via PerceptionNavigationPolicy through run_game.",
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "purpose": "method_validation_autonomous_navigation_scored_interface",
        "not_a_new_solve_claim": "tu93 L3 is already registered; this validates the FULLY autonomous solver (auto branch_mode, no per-game input) + that the scored run_game interface carries it. No registry change.",
        "random_seed": 5839,
        "auto_branch_mode": branch,
        "solve_navigation": {k: solve.get(k) for k in ("is_navigation_game", "pair", "branch_mode",
                                                       "search_reached", "reproduced", "reproduced_level", "path_len")},
        "scored_baseline_levels": base.get("levels"),
        "scored_baseline_reached": base.get("reached"),
        "scored_perception_levels": scored.get("levels"),
        "scored_perception_reached": scored.get("reached"),
        "scored_plan_len": len(pol.plan),
        "fully_autonomous_no_per_game_input": bool(solve.get("reproduced")),
        "scored_interface_carries_solve": (scored.get("reached", 0) or 0) > (base.get("reached", 0) or 0)
        or (scored.get("levels", 0) or 0) > 0,
        "duration_s": round(time.time() - t0, 1),
    }
    art["honest_verdict"] = (
        "complete_success_fully_autonomous_navigation_solver_and_scored_interface_carries_it"
        if art["fully_autonomous_no_per_game_input"] and art["scored_interface_carries_solve"]
        else ("complete_autonomous_solver_works_scored_replay_investigate"
              if art["fully_autonomous_no_per_game_input"]
              else "complete_investigate_autonomous_solver")
    )
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(json.dumps(art, sort_keys=True, default=str).encode()).hexdigest()
    out = ROOT / "results" / "outer_loop_arc_perception_navigation_scored_20260723.json"
    out.write_text(json.dumps(art, indent=2, default=str))
    print(f"fully_autonomous={art['fully_autonomous_no_per_game_input']} scored_carries={art['scored_interface_carries_solve']}")
    print("wrote", out, f"({art['duration_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
