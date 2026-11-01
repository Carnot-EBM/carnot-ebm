"""REQ-REPORT-8694: derive finite controls without promoting absent experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import v745_capstone_inputs as e
from carnot.reporting import arc_observer_transfer_8678 as arc
from carnot.reporting import spline_precision_study_8692 as precision

Json = dict[str, Any]


def reduce(work: Json) -> Json:
    """Re-read sealed per-update and episode rows; unexecuted branches have no metric."""
    history = e.read(work["history_reference"])
    old = e.read(history["primitive_reference"])
    producer = next(r for r in old["slots"] if r["experiment_id"] == 8678)
    panel = e.read(producer["primary_reference"])
    panel_work = e.read(panel["work_reference"])
    episodes = [e.read(ref) for ref in panel_work["references"]]
    horizons = arc.reduce(episodes)
    if any(panel[k] != v for k, v in horizons.items()):
        raise ValueError("original_arc_reduction")
    values = {
        str(r["experiment_id"]): e.read(r["primary_reference"])
        for r in work["slots"]
        if r["producer_executed"]
    }
    numerical = values["8692"]
    primitives = e.read(numerical["primitive_reference"])
    if (
        precision.build(
            primitives, numerical["validation_receipts"], Path(numerical["publication_output"])
        )
        != numerical
    ):
        raise ValueError("numerical_producer_construction")
    points = primitives["precision_rows"] + primitives["trajectory_rows"]
    numeric_rows = []
    for bits, fractional in ((16, 12), (24, 18), (32, 24)):
        selected = [r for r in points if r["bits"] == bits]
        expected = {
            (h, seq, i)
            for h in range(5)
            for seq, count in [
                ("one_step", 512),
                *[(k, 256) for k in sorted({r["sequence"] for r in points} - {"one_step"})],
            ]
            for i in range(count)
        }
        keys = [(r["head_index"], r["sequence"], r["update_index"]) for r in selected]
        if len(keys) != 7680 or len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError("numerical_row_inventory")
        summaries = [r for r in primitives["rows"] if r["bits"] == bits]
        saturation = sum(r["saturation"] for r in summaries)
        overflow = sum(r["overflow"] for r in summaries)
        maximum_error = max(r["probability_error"] for r in selected)
        parity = all(
            r["decision"] == r["float64_decision"] and r["decision_parity"] for r in selected
        )
        finite = all(r["finite"] for r in selected)
        writes = max(r["local_writes"] for r in selected)
        numeric_rows.append(
            dict(
                bits=bits,
                fractional=fractional,
                row_count=len(selected),
                independent_questions=0,
                maximum_probability_error=maximum_error,
                saturation=saturation,
                overflow=overflow,
                typed_decision_parity=parity,
                finite=finite,
                maximum_local_writes=writes,
                fallback_count=sum(bool(r["fallback"]) for r in selected),
                passed=finite
                and maximum_error <= 0.001
                and parity
                and writes <= 37
                and saturation == overflow == 0,
                physical_execution=False,
                scope="constructed CPU integer emulation",
            )
        )
    checkpoint = precision.authenticate()
    if checkpoint["failures"]:
        raise ValueError("numerical_checkpoint_authentication")
    hardware = values["8693"]
    return dict(
        precision=numeric_rows,
        numerical_checkpoint_references=checkpoint["refs"],
        arc=dict(
            complete_horizon_count=horizons["complete_horizon_count"],
            censored_count=len(horizons["censored_reason_rows"]),
            independent_games=len(horizons["per_game_results"]),
            per_game_results=horizons["per_game_results"],
            horizon_rows=horizons["horizon_rows"],
            efficacy_estimate=None,
            decision="stop_closed_V744_diagnostic_retain_policy",
            v745_ledger_decision=dict(
                honest_verdict=values["8691"]["honest_verdict"],
                flagged_adversarial=values["8691"]["flagged_adversarial"],
                admitted=False,
                policy_change_authorized=False,
            ),
        ),
        learning=[
            dict(
                measured=False,
                future_brier=None,
                joint_retention=None,
                missing_reason="Exp8682/8683/8685/8686 producer execution absent",
            )
        ],
        semantics=[
            dict(
                measured=False,
                lexical_collisions=None,
                source_to_record_errors=None,
                constructed_truth_is_independent_generalization=False,
                missing_reason="Exp8684/8687/8690 producer execution absent",
            )
        ],
        deployment=[
            dict(
                measured=False,
                same_batch_cost_ratio=None,
                nfr01_score=0,
                missing_reason="Exp8688/8689 producer execution absent",
            )
        ],
        hardware={
            k: hardware[k]
            for k in [
                "per_board_results",
                "operation_support_rows",
                "conditional_acceleration_bounds",
                "hardware_100x_requirements",
                "board_reopen_requirements",
            ]
        },
        history={
            k: history[k]
            for k in [
                "historical_quarantines",
                "preserved_nulls",
                "preserved_v737_supervisor_null",
                "preserved_v743_batch_one_null",
                "historical_stamped_outcomes",
            ]
        },
        literature=work["protocol"]["literature_method_rows"],
    )
