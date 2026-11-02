"""Experiment 5602: matched-budget A/B for InertClickSigPruner
(REQ-ARC-FCP-5595-2) -- the pending flip-decision measurement task 9's own
wiring follow-on named: "flipping [SUBMITTED_INERT_CLICK_PRUNER_ENABLED] on
for the SCORED agent needs its own matched-budget offline A/B (states/
actions-expanded reduction, zero regression in reproduced levels) first, per
the solve_rate_dropped guardrail."

Mirrors HazardMovePruner's own tu93 A/B precedent exactly (same mechanism:
`OfflineSolver(move_pruner=...)`, same `states_expanded` metric already
tracked by `OfflineSolver.last_states_expanded`, same correctness backstop --
`arc_solver_kit.reproduce`, the offline reproduction gate). Unlike tu93
(keyboard-maze, no clicks), this A/B needs a CLICK-heavy game with a
registered `GameAdapter` reachable via `OfflineSolver.solve_level` -- `m0r0`
(confirmed click-heavy by exp5595; L1-L2 solved via `OfflineSolver` per
`ops/arc_solve_registry.yaml`).

Runs `scripts/arc_loop_solve.solve_adaptered` TWICE at the SAME
`--target-level` with hazard_prune/mask_prune held FIXED at False in both
arms (isolating the inert-click-prune variable), varying ONLY
`inert_click_prune`. Uses the pruner's REAL, already-validated default
parameters (`min_observations=4`, `min_specificity=0.9`) -- not tuned
favorably for this measurement, per this project's own rigor discipline.

This is a measurement script, not a live-path parameter flip:
`SUBMITTED_INERT_CLICK_PRUNER_ENABLED` stays `False` regardless of this
script's result. `solve_provenance` stays `development_proxy`.

Spec refs: REQ-ARC-FCP-5595-2, SCENARIO-ARC-FCP-5595-2-MATCHED-BUDGET-AB.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(SCRIPTS_ROOT))

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5602_inert_click_pruner_matched_budget_ab"
RESULT_RELATIVE_PATH = "results/experiment_5602_inert_click_pruner_matched_budget_ab.json"
SCHEMA = "carnot.exp5602.inert_click_pruner_matched_budget_ab.v1"
INFERENCE_SUBSTRATE = "offline_arc_sim_no_quota"
RANDOM_SEED = 5602
DEFAULT_GAME = "m0r0"
DEFAULT_TARGET_LEVEL = 2

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "game",
    "target_level",
    "baseline",
    "treatment",
    "states_expanded_reduction",
    "reduction_pct",
    "gate_definition",
    "live_wired_supplementary_check",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; reports whichever outcome the matched-budget A/B "
        "actually produced (helps / no-op / regresses) -- a zero-reduction honest null is a "
        "valid, informative result at this search budget, not a failure requiring escalation, "
        "per exp5595's own precedent (0 confidently-inert signatures at that budget)"
    },
    "baseline": {
        "principle": "hazard_prune=False, mask_prune=False, inert_click_prune=False -- isolates "
        "the inert-click-prune variable from any other pruner"
    },
    "treatment": {
        "principle": "identical config with inert_click_prune=True -- the pruner's REAL, "
        "already-validated default parameters (min_observations=4, min_specificity=0.9), not "
        "tuned favorably for this measurement"
    },
    "states_expanded_reduction": {
        "principle": "baseline.states_expanded - treatment.states_expanded; the load-bearing "
        "efficiency claim, mirroring HazardMovePruner's tu93 A/B (states_expanded 2947 -> 2859)"
    },
    "gate_definition": {
        "principle": "the reproduction gate (arc_solver_kit.reproduce) is the correctness "
        "backstop for BOTH arms -- a states_expanded reduction only counts as a real win if "
        "both arms reach the SAME target level AND both pass offline_reproduced=True"
    },
    "live_wired_supplementary_check": {
        "principle": "OfflineSolver's directed, verifier-guided search may simply never "
        "exercise repeated inert clicks the way broad live exploration does -- this second, "
        "independent measurement wires inert_click_pruner=True into a REAL E3AgentPolicy "
        "exploration run (matching exp5595's own construction) and reports the pruner's own "
        "stats() at the end, confirming whether the wiring engages at all outside the "
        "OfflineSolver harness, not just whether it changes states_expanded there"
    },
    "solve_provenance": {
        "principle": "development_proxy -- a prototype/measurement script, not a live-path flip"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
}


def preconditions(root: Path = REPO_ROOT) -> JsonDict:
    checks: dict[str, bool] = {}
    try:
        from carnot.agentic import arc_game_adapters as adapters

        checks["adapter_registered"] = adapters.get_adapter(DEFAULT_GAME) is not None
    except Exception:
        checks["adapter_registered"] = False
    try:
        from carnot.agentic.arc_inert_click_pruner import InertClickSigPruner  # noqa: F401

        checks["inert_click_pruner_import"] = True
    except Exception:
        checks["inert_click_pruner_import"] = False
    try:
        from arc_loop_solve import solve_adaptered  # noqa: F401

        checks["solve_adaptered_import"] = True
    except Exception:
        checks["solve_adaptered_import"] = False
    try:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        env = arc.make(DEFAULT_GAME, scorecard_id=arc.open_scorecard())
        env.reset()
        checks["offline_arcade_makes_env"] = True
    except Exception:
        checks["offline_arcade_makes_env"] = False
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(preconds: JsonDict) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if not value:
            return key
    return None


def _checksum(payload: JsonDict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _run_arm(*, game: str, target_level: int, inert_click_prune: bool) -> JsonDict:
    from arc_loop_solve import solve_adaptered

    out = solve_adaptered(
        game,
        target_level,
        hazard_prune=False,
        mask_prune=False,
        inert_click_prune=inert_click_prune,
    )
    return {
        "reached_level": int(out["reached_level"]),
        "states_expanded": int(out["states_expanded"]),
        "offline_reproduced": bool(out["offline_reproduced"]),
        "inert_click_prune": bool(inert_click_prune),
        "pruner_stats": out.get("hazard_pruner_stats"),
    }


def _live_wired_check(*, game: str, explore_budget: int, live_total_budget: int) -> JsonDict:
    """Wire inert_click_pruner=True into a REAL E3AgentPolicy exploration run (matching
    exp5595's own real-game construction, no explicit LLM proposer needed since classical
    salience-driven exploration never invokes it) and report the pruner's real engagement --
    independent of whether OfflineSolver's directed search ever exercised it."""

    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    policy = E3AgentPolicy(
        game, proposer=None, explore_budget=explore_budget, inert_click_pruner=True
    )
    lb.run_game(game, policy, budget=live_total_budget)
    stats = policy.explorer.inert_click_pruner.stats()
    return {
        "transitions_collected": len(policy.transitions),
        "pruner_stats": stats,
    }


def build_artifact(
    *,
    game: str = DEFAULT_GAME,
    target_level: int = DEFAULT_TARGET_LEVEL,
    explore_budget: int = 6,
    live_total_budget: int = 40,
    root: Path = REPO_ROOT,
) -> JsonDict:
    started_at = time.time()
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)
    if miss:
        artifact: JsonDict = {
            "experiment": EXPERIMENT_ID,
            "schema": SCHEMA,
            "result_path": RESULT_RELATIVE_PATH,
            "honest_verdict": f"complete: blocked_{miss}",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "field_principles": FIELD_PRINCIPLES,
            "verifier_is_oracle": False,
            "game": game,
            "target_level": int(target_level),
            "baseline": {},
            "treatment": {},
            "states_expanded_reduction": 0,
            "reduction_pct": 0.0,
            "gate_definition": (
                "reduction only counts if both arms reach the same target level and both pass "
                "the offline reproduction gate"
            ),
            "live_wired_supplementary_check": {},
            "solve_provenance": "development_proxy",
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "duration_s": round(time.time() - started_at, 3),
            "preconditions_checked": preconds,
        }
        artifact["reproducibility_checksum"] = _checksum(
            {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
        )
        return artifact

    baseline = _run_arm(game=game, target_level=target_level, inert_click_prune=False)
    treatment = _run_arm(game=game, target_level=target_level, inert_click_prune=True)
    try:
        live_check = _live_wired_check(
            game=game, explore_budget=explore_budget, live_total_budget=live_total_budget
        )
    except Exception as exc:
        live_check = {"error": repr(exc)[:200]}

    both_reproduced = bool(baseline["offline_reproduced"] and treatment["offline_reproduced"])
    same_level_reached = baseline["reached_level"] == treatment["reached_level"]
    reduction = int(baseline["states_expanded"] - treatment["states_expanded"])
    reduction_pct = round(100.0 * reduction / float(max(baseline["states_expanded"], 1)), 3)
    live_pruned = int(
        (live_check.get("pruner_stats") or {}).get("pruned", 0) if "error" not in live_check else 0
    )

    if (
        not both_reproduced
        or not same_level_reached
        or treatment["reached_level"] < baseline["reached_level"]
    ):
        verdict = "complete: inert_click_pruner_ab_regressed_reproduction_or_level"
    elif reduction > 0 or live_pruned > 0:
        verdict = (
            f"complete: inert_click_pruner_ab_reduces_states_expanded_{reduction}"
            f"_live_pruned_{live_pruned}_on_{game}"
        )
    else:
        verdict = f"complete: inert_click_pruner_ab_no_op_offline_and_live_at_this_budget_on_{game}"

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "game": game,
        "target_level": int(target_level),
        "baseline": baseline,
        "treatment": treatment,
        "states_expanded_reduction": reduction,
        "reduction_pct": reduction_pct,
        "gate_definition": (
            "reduction only counts if both arms reach the same target level and both pass "
            "the offline reproduction gate"
        ),
        "live_wired_supplementary_check": live_check,
        "solve_provenance": "development_proxy",
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.time() - started_at, 3),
        "preconditions_checked": preconds,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(
        {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
    )
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper, exercised manually
    artifact = build_artifact()
    out_path = REPO_ROOT / RESULT_RELATIVE_PATH
    out_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path} -- honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
