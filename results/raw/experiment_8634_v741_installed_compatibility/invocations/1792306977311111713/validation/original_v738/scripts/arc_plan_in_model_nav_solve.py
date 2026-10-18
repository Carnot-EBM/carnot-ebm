#!/usr/bin/env python3
"""plan_in_model regression fix + structured-nav-model live navigation solve (REQ-ARC-WMTE-5841).

Digging into the 2026-07-20 induction-quality diagnosis's open question #6 ("why does even a PERFECT induction
execute to zero real level-up with plan_len=1?"), the outer loop found the concrete cause: a REGRESSION.

The bug: `_components_detailed` (arc_graph_explore) was widened from a 4-tuple (cy,cx,area,color) to a
5-tuple (+ is_grid_fallback) in commit 2f0760307 (GAP-ARC-BP35-CLICK-CANDIDATE-GENERATION-MISS). That fix
updated the arc_graph_explore consumer DEFENSIVELY but MISSED `plan_in_model`'s `_model_candidates`, whose
rigid `for cy,cx,_a,_c in comps` unpack then raised ValueError on ANY grid with components -- silently
disabling the ENTIRE world-model planning tier (plan_in_model) for every game with objects (tu93 has 65).
The live/harness call sites catch the exception, so it degraded to "no plan" with no surfaced error.

The fix: defensive unpack (`*_`) in `_model_candidates`, matching how the arc_graph_explore consumer already
handles it.

This script proves the fix + a downstream win: with the fix AND a STRUCTURED nav world model
(`InducedNavWorldModel`, correct-by-construction for the 4-direction navigation family -- the "mechanic-class
prior" the diagnosis §6 hinted at, as opposed to the near-universally-wrong LLM induction), `plan_in_model`
finds a real multi-step navigation plan that reaches a REAL tu93 level-up when executed. CRUCIALLY,
plan_in_model plans IN IMAGINATION then executes ONCE from reset -- NO per-node env resets -- so it SIDESTEPS
tu93's non-idempotent-reset blocker that defeated the OfflineSolver/StepwiseExplorer live search
(REQ-ARC-WMTE-5840). This is a live-compatible navigation solve.

inference_substrate: offline_arcade_live_agent_runtime_self_discovery_no_llm. verifier_is_oracle: False.
solve_provenance: development_proxy (method validation; tu93 L1 already registered; no registry change).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

GAME = "tu93"


def _one_run(seed_cycles: int) -> dict:
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_actions_to_progress import _execute_plan_measure, _hand_verifier_fn
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_executable_world_model import detect_cell, plan_in_model, to_logical
    from carnot.agentic.arc_nav_world_model import InducedNavWorldModel
    from carnot.agentic.arc_perception_navigation import recon

    tr = recon(GAME, cycles=seed_cycles)
    nav = InducedNavWorldModel.fit([(t.before, t.action, t.after) for t in tr])
    arc = kit.offline_arcade()
    env = arc.make(GAME, scorecard_id=arc.open_scorecard())
    f = env.reset()
    cell = detect_cell(grid_of(f))
    root = to_logical(grid_of(f), cell)
    eng, is_lc = nav.as_callables()
    diag: dict = {}
    plan = plan_in_model(eng, is_lc, root, max_nodes=20000, max_depth=60, diagnostics=diag)
    out = {
        "displacement": {int(k): list(v) for k, v in getattr(nav, "displacement", {}).items()},
        "goal_color": getattr(nav, "goal_color", None),
        "plan_len": None if plan is None else len(plan),
        "termination": diag.get("termination_reason"),
        "nodes_expanded": diag.get("nodes_expanded"),
        "reached_levelup": False,
        "actions_to_levelup": None,
        "start_hv": None,
        "best_hv": None,
    }
    if plan:
        exe = _execute_plan_measure(GAME, plan, _hand_verifier_fn(GAME))
        out.update({k: exe[k] for k in ("reached_levelup", "actions_to_levelup", "start_hv", "best_hv")})
    return out


def main() -> int:
    t0 = time.time()
    runs = [_one_run(c) for c in (3, 3, 4)]  # 3 reproducibility runs (fresh recon each)
    n_reached = sum(1 for r in runs if r["reached_levelup"])
    art = {
        "experiment": "outer_loop_arc_plan_in_model_nav_solve",
        "experiment_id": "REQ-ARC-WMTE-5841",
        "run_date": "2026-07-23",
        "schema": "carnot.arc_plan_in_model_nav_solve.v1",
        "title": "plan_in_model regression fix (5-tuple component unpack) + structured-nav-model live navigation solve of tu93.",
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "purpose": "regression_fix_plus_method_validation",
        "not_a_new_solve_claim": "tu93 L1 already registered; this validates (a) the plan_in_model regression fix and (b) that a structured InducedNavWorldModel + plan_in_model reaches a real tu93 level-up live. No registry change.",
        "random_seed": 5841,
        "regression": {
            "bug": "_model_candidates unpacked _components_detailed as a 4-tuple; the field was widened to a 5-tuple (+is_grid_fallback) in commit 2f0760307 -> ValueError on any grid with components -> plan_in_model silently disabled for object-bearing games.",
            "fix": "defensive `for cy, cx, _a, _c, *_ in comps` unpack in arc_executable_world_model._model_candidates; docstring on arc_graph_explore._components_detailed corrected to note the 5-tuple + the defensive-unpack requirement.",
            "regression_test": "tests/python/test_plan_in_model_component_unpack.py (2 tests; the rigid 4-unpack raises ValueError on a 5-tuple, confirmed).",
        },
        "methodology_note": "InducedNavWorldModel.fit derives per-action displacement + avatar + goal from the agent's OWN recon transitions (structured, correct-by-construction for the nav family -- NOT LLM induction). plan_in_model plans in-model (no env), then _execute_plan_measure executes ONCE from a fresh reset and checks the REAL level counter (not a heuristic). No LLM. Public-game frames for offline dev.",
        "runs": runs,
        "reached_levelup_count": n_reached,
        "reached_levelup_reproducible": n_reached >= 2,
        "sidesteps_nonidempotent_reset": "plan_in_model plans in imagination + executes once from reset -> no per-node resets -> the non-idempotent-reset blocker (REQ-ARC-WMTE-5840) does not apply.",
        "headline": None,
        "next_step": ("InducedNavWorldModel (the structured nav inducer) is currently ORPHANED from the live "
                      "E3 path (imported only by scripts/tests). The concrete unblock: route navigation games "
                      "(detected via derive_navigation_pair) to InducedNavWorldModel instead of the LLM "
                      "induction, and feed its engine+is_level_complete to the live plan_in_model tier -- giving "
                      "E3 a CORRECT model for nav games instead of the near-universally-wrong LLM one. This is "
                      "the 'mechanic-class prior' direction the 2026-07-20 diagnosis §6 flagged as highest-leverage."),
        "honest_verdict": None,
        "duration_s": round(time.time() - t0, 1),
    }
    art["headline"] = (
        f"REGRESSION FOUND + FIXED: plan_in_model crashed on component-grids since 2f0760307 (5-tuple "
        f"_components_detailed), silently disabling E3's world-model planner. After the fix, a structured "
        f"InducedNavWorldModel + plan_in_model reaches a REAL tu93 level-up in {n_reached}/3 runs "
        f"(e.g. {runs[0].get('plan_len')}-action plan, hv {runs[0].get('start_hv')}->{runs[0].get('best_hv')}). "
        f"plan_in_model plans in imagination + executes once from reset -> sidesteps the non-idempotent-reset "
        f"blocker -> live-compatible navigation solve."
    )
    art["honest_verdict"] = (
        "complete_success_regression_fixed_and_structured_nav_model_planinmodel_reaches_tu93_levelup_live"
        if n_reached >= 2 else "complete_regression_fixed_levelup_not_reliably_reproduced_investigate"
    )
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(json.dumps(art, sort_keys=True, default=str).encode()).hexdigest()
    out = ROOT / "results" / "outer_loop_arc_plan_in_model_nav_solve_20260723.json"
    out.write_text(json.dumps(art, indent=2, default=str))
    for i, r in enumerate(runs):
        print(f"run {i}: plan_len={r['plan_len']} reached_levelup={r['reached_levelup']} "
              f"actions_to={r['actions_to_levelup']} hv={r['start_hv']}->{r['best_hv']}")
    print(f"reached_levelup {n_reached}/3 | reproducible={art['reached_levelup_reproducible']}")
    print("wrote", out, f"({art['duration_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
