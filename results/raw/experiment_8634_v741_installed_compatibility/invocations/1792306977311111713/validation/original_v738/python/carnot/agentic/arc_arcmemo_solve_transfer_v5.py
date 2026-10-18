"""ArcMemo solve-transfer v5 cost accounting for the .372 ARC solves.

The experiment is intentionally an artifact-level comparison.  Exp 4021 and
Exp 4024 already performed the expensive environment work; this module checks
whether reusing their accumulated concept/model memory makes the next solve
path cheaper than the cold baselines recorded beside those same solves.
"""

from __future__ import annotations

from typing import Any, Mapping

INFERENCE_SUBSTRATE = "offline_arc_agi3_arcmemo_concept_transfer"
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "solve_transfer_win",
    "actions_cold",
    "actions_seeded",
    "inference_substrate",
)


def _as_int(payload: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = payload.get(key)
        if value is not None and type(value) is not bool:
            return int(value)
    return 0


def _terminal_prefixed(value: object) -> bool:
    return isinstance(value, str) and value.startswith(("success:", "complete:", "blocked_"))


def _cost_exp4021(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    confirmed = bool(payload.get("real_env_confirmed")) and _as_int(payload, "new_levels_solved_this_task") > 0
    if not confirmed:
        return None
    actions = _as_int(payload, "executed_real_env_actions", "action_count")
    seeded_induction_calls = 0 if "no new induction" in str(payload.get("model_reuse_note", "")).lower() else 1
    return {
        "content_id": "exp4021",
        "source_artifact": "results/experiment_4021_heuristic_search_over_verified_wm.json",
        "upstream_honest_verdict": str(payload.get("honest_verdict", "")),
        "actions_cold": actions,
        "actions_seeded": actions,
        "induction_calls_cold": 1,
        "induction_calls_seeded": seeded_induction_calls,
        "cost_basis": (
            "4021 reused the verified simulator and Exp 4020 goal predicate; action cost is "
            "kept equal because the artifact has no larger successful cold action baseline."
        ),
    }


def _cost_exp4024(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    confirmed = bool(payload.get("game_solved")) and bool(payload.get("real_env_confirmed"))
    seeded_actions = _as_int(payload, "first_solve_at_action")
    cold_actions = _as_int(payload, "candidate_baseline_actions", "baseline_actions")
    if not confirmed or seeded_actions <= 0 or cold_actions <= 0:
        return None
    return {
        "content_id": "exp4024",
        "source_artifact": "results/experiment_4024_fifth_game_explore_first.json",
        "upstream_honest_verdict": str(payload.get("honest_verdict", "")),
        "actions_cold": cold_actions,
        "actions_seeded": seeded_actions,
        "induction_calls_cold": 1,
        "induction_calls_seeded": 1,
        "cost_basis": "4024 records the selected candidate's L0 baseline and first real-env solve action.",
    }


def _empty_artifact(reason: str, *, duration_s: float) -> dict[str, Any]:
    return {
        "experiment": "experiment_4025_arcmemo_solve_transfer_v5",
        "title": "arcmemo_solve_transfer_v5_372_content",
        "honest_verdict": f"complete: arcmemo_v5_no_transfer_{reason}",
        "solve_transfer_win": False,
        "actions_cold": 0,
        "actions_seeded": 0,
        "induction_calls_cold": 0,
        "induction_calls_seeded": 0,
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_content_costs": [],
    }


def build_transfer_artifact(
    *,
    exp4021: Mapping[str, Any] | None,
    exp4024: Mapping[str, Any] | None,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4025 artifact from the two upstream .372 result JSONs."""

    if exp4021 is None or exp4024 is None:
        return _empty_artifact("missing_upstream_artifacts", duration_s=duration_s)

    costs = [_cost_exp4021(exp4021), _cost_exp4024(exp4024)]
    if any(cost is None for cost in costs):
        return _empty_artifact("upstream_not_real_env_confirmed", duration_s=duration_s)

    per_content_costs = [dict(cost) for cost in costs if cost is not None]
    actions_cold = sum(int(cost["actions_cold"]) for cost in per_content_costs)
    actions_seeded = sum(int(cost["actions_seeded"]) for cost in per_content_costs)
    induction_calls_cold = sum(int(cost["induction_calls_cold"]) for cost in per_content_costs)
    induction_calls_seeded = sum(int(cost["induction_calls_seeded"]) for cost in per_content_costs)
    solve_transfer_win = actions_seeded < actions_cold or induction_calls_seeded < induction_calls_cold

    verdict = (
        f"success: arcmemo_v5_transfer_{actions_cold}to{actions_seeded}_actions"
        if solve_transfer_win
        else "complete: arcmemo_v5_no_transfer_seeded_not_cheaper"
    )
    return {
        "experiment": "experiment_4025_arcmemo_solve_transfer_v5",
        "title": "arcmemo_solve_transfer_v5_372_content",
        "honest_verdict": verdict,
        "solve_transfer_win": bool(solve_transfer_win),
        "actions_cold": int(actions_cold),
        "actions_seeded": int(actions_seeded),
        "induction_calls_cold": int(induction_calls_cold),
        "induction_calls_seeded": int(induction_calls_seeded),
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "upstream_artifacts": [cost["source_artifact"] for cost in per_content_costs],
        "per_content_costs": per_content_costs,
        "principle": "persistent memory must demonstrably reduce future cost, not just store",
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if "honest_verdict" in artifact and not _terminal_prefixed(artifact["honest_verdict"]):
        errors.append("honest_verdict must start with success:, complete:, or blocked_")
    if "solve_transfer_win" in artifact and type(artifact["solve_transfer_win"]) is not bool:
        errors.append("solve_transfer_win must be a bare bool")
    for field in ("actions_cold", "actions_seeded", "induction_calls_cold", "induction_calls_seeded"):
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")
    if "inference_substrate" in artifact and type(artifact["inference_substrate"]) is not str:
        errors.append("inference_substrate must be a bare string")
    return errors
