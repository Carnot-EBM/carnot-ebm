"""Conservative stop policy for HardNet++/DSP continuous repair replay.

Spec: REQ-KONA-031, SCENARIO-KONA-031
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

REQUIRED_REPLAY_FIELDS = (
    "case_id",
    "cohort",
    "before_violation_energy",
    "before_violation_count",
    "after_violation_energy",
    "channel_score",
    "repair_helped",
    "predicted_continue",
)


def _normalise_replay_row(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_REPLAY_FIELDS if field not in row]
    if missing:
        raise ValueError(f"missing replay row fields: {', '.join(missing)}")

    numeric_fields = {
        "before_violation_energy": float(row["before_violation_energy"]),
        "before_violation_count": float(row["before_violation_count"]),
        "after_violation_energy": float(row["after_violation_energy"]),
        "channel_score": float(row["channel_score"]),
    }
    if any(not math.isfinite(value) for value in numeric_fields.values()):
        raise ValueError("replay row numeric values must be finite")
    if any(value < 0.0 for value in numeric_fields.values()):
        raise ValueError("replay row numeric values must be non-negative")

    return {
        "case_id": str(row["case_id"]),
        "cohort": str(row["cohort"]),
        "before_violation_energy": numeric_fields["before_violation_energy"],
        "before_violation_count": int(numeric_fields["before_violation_count"]),
        "after_violation_energy": numeric_fields["after_violation_energy"],
        "channel_score": numeric_fields["channel_score"],
        "repair_helped": bool(row["repair_helped"]),
        "predicted_continue": bool(row["predicted_continue"]),
    }


def _is_residual_nonlinear_local_linear(cohort: str) -> bool:
    return cohort.startswith("exp1291_") and cohort.endswith("_local_linear")


def _decision_for_row(
    row: Mapping[str, Any],
    *,
    threshold: float,
    help_energy_tolerance: float,
) -> dict[str, Any]:
    hard_violations_remain = bool(
        row["before_violation_count"] > 0
        or row["before_violation_energy"] > help_energy_tolerance
    )
    residual_local_linear = _is_residual_nonlinear_local_linear(row["cohort"])
    if not hard_violations_remain:
        conservative_continue = False
        stop_reason = "hard_feasible"
    elif residual_local_linear:
        conservative_continue = False
        stop_reason = "residual_nonlinear_local_linear"
    elif row["channel_score"] < threshold:
        conservative_continue = False
        stop_reason = "below_feasibility_threshold"
    else:
        conservative_continue = True
        stop_reason = "continue"

    return {
        **row,
        "hard_violations_remain": hard_violations_remain,
        "residual_nonlinear_local_linear": residual_local_linear,
        "conservative_continue": conservative_continue,
        "stop_reason": stop_reason,
    }


def _residual_nonlinear_cases(decisions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cohorts: dict[str, list[Mapping[str, Any]]] = {}
    for row in decisions:
        if row["stop_reason"] == "residual_nonlinear_local_linear":
            cohorts.setdefault(row["cohort"], []).append(row)

    residuals: list[dict[str, Any]] = []
    for cohort in sorted(cohorts):
        rows = cohorts[cohort]
        residuals.append(
            {
                "cohort": cohort,
                "count": len(rows),
                "example_case_ids": [str(row["case_id"]) for row in rows[:3]],
                "mean_after_violation_energy": sum(
                    float(row["after_violation_energy"]) for row in rows
                )
                / len(rows),
                "action": "stop_local_linear_and_route_to_hardnetpp",
            }
        )
    return residuals


def evaluate_stop_policy(
    replay_rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.5,
    help_energy_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Replay the conservative HardNet++/DSP stop policy.

    **Researcher summary:**
        The policy accepts the DSP channel as a useful signal only after a hard
        guard: stop when hard feasibility has already been reached, and stop
        nonlinear local-linear residual cases because Exp 1291 showed HardNet++
        is the viable repair route for that geometry.

    Spec: REQ-KONA-031, SCENARIO-KONA-031
    """
    rows = list(replay_rows)
    if not rows:
        raise ValueError("at least one replay row is required")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if help_energy_tolerance < 0.0:
        raise ValueError("help_energy_tolerance must be non-negative")

    decisions = [
        _decision_for_row(
            _normalise_replay_row(row),
            threshold=threshold,
            help_energy_tolerance=help_energy_tolerance,
        )
        for row in rows
    ]

    conservative_continue = [
        row for row in decisions if row["conservative_continue"]
    ]
    conservative_stop = [
        row for row in decisions if not row["conservative_continue"]
    ]
    true_continue = [row for row in conservative_continue if row["repair_helped"]]
    false_continue = [
        row for row in conservative_continue if not row["repair_helped"]
    ]
    true_stop = [row for row in conservative_stop if not row["repair_helped"]]
    false_stop = [row for row in conservative_stop if row["repair_helped"]]
    dsp_continue = [row for row in decisions if row["predicted_continue"]]
    dsp_true_continue = [row for row in dsp_continue if row["repair_helped"]]

    return {
        "candidate_transitions": len(decisions),
        "baseline_dsp_continue_precision": (
            len(dsp_true_continue) / len(dsp_continue) if dsp_continue else 0.0
        ),
        "conservative_continue_recommendations": len(conservative_continue),
        "conservative_stop_recommendations": len(conservative_stop),
        "true_continue_recommendations": len(true_continue),
        "false_continue_recommendations": len(false_continue),
        "policy_true_stop_recommendations": len(true_stop),
        "policy_false_stop_recommendations": len(false_stop),
        "policy_stop_accuracy": (
            len(true_stop) / len(conservative_stop) if conservative_stop else 0.0
        ),
        "stop_policy_precision": (
            len(true_continue) / len(conservative_continue)
            if conservative_continue
            else 0.0
        ),
        "residual_nonlinear_cases": _residual_nonlinear_cases(decisions),
        "per_case": decisions,
    }
