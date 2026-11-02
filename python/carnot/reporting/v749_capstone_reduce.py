"""REQ-REPORT-8748: absence and failed execution cannot supply numerical science."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import equal_budget_verifier_8729 as verifier
from carnot.reporting import v746_capstone_inputs as io
from carnot.reporting.v747_capstone_reduce import same

Json = dict[str, Any]
HISTORY = dict(
    path=str(io.ROOT / "results/experiment_8736_v748_capstone.json"),
    sha256="sha256:0920d4fc3ce9c76c2287a8b6eb43a548fee95dba85eccadd834c879fcd7c081e",
)
COMPARISONS = {
    8740: (
        "learning",
        "learning_rows_ready_score",
        "future Brier and atomic delayed-feedback updates",
    ),
    8741: (
        "retention",
        "learning_benefit_ready_score",
        "future proper loss and simultaneous retention",
    ),
    8742: (
        "verifier_transfer",
        "verifier_transfer_ready_score",
        "frozen five-call new-family detection",
    ),
    8744: (
        "feature_reuse_cost",
        "installed_cost_ready_score",
        "matched B16 complete parent lifetime throughput",
    ),
    8745: (
        "bounded_durable_cost",
        "bounded_update_cost_ready_score",
        "three32-update windows;90 processes;2880 durable updates",
    ),
}


def reduce(work: Json) -> Json:
    """Reconstruct admitted historical units and refuse to invent missing current comparisons."""
    slots = {r["experiment_id"]: r for r in work["slots"]}
    if any(slots[n]["producer_qualified"] for n in COMPARISONS):
        raise ValueError("changed_prerequisite_new_admitted_comparison_requires_its_frozen_reducer")
    history = io.load(HISTORY)
    historical_verifier = next(r for r in history["rows"] if r["experiment_id"] == 8729)
    value = io.load(historical_verifier["primary_reference"])
    primitive = io.load(value["primitive_reference"])
    summary = verifier.summarize(primitive["frozen"], primitive["measurement"]["rows"])
    for key in ["arm_rows", "paired_world_rows", "paired_intervals", "probe_rows"]:
        same(summary[key], value[key], "preserved_v748_" + key)
    gates, blocked = [], {}
    for number, (name, field, scope) in COMPARISONS.items():
        row = slots[number]
        gate = io.observe(row["declared_reference"], row["task_id"], field, 1)
        gate.update(
            check="admitted_comparison", producer_qualified=row["producer_qualified"], passed=False
        )
        gates.append(gate)
        blocked[name] = dict(
            measured=False,
            verdict_class="blocked",
            comparison=None,
            scope=scope,
            producer_executed=row["producer_executed"],
            primary_reference=row["primary_reference"],
            admission_gate=gate,
            rows=[],
            independent_count=0,
            model_calls=0,
            physical_execution=False,
        )
    arc: Json = dict(
        measured=False,
        new_event_count=None,
        generalization_measured=False,
        policy_change_authorized=False,
    )
    if slots[8746]["producer_qualified"]:
        value = io.load(slots[8746]["primary_reference"])
        primitive = io.load(value["primitive_reference"])
        same(primitive["new_event_rows"], value["new_event_rows"], "arc_new_event_rows")
        same(primitive["frontier_before"], value["frontier_before"], "arc_frontier_before")
        same(primitive["frontier_after"], value["frontier_after"], "arc_frontier_after")
        arc.update(
            measured=True,
            new_event_count=len(primitive["new_event_rows"]),
            frontier_unchanged=primitive["frontier_before"] == primitive["frontier_after"],
            verdict_class=value["verdict_class"],
            closed_panel_replayed=False,
            reference=slots[8746]["primary_reference"],
        )
    diagnostics = []
    for number in [8737, 8738, 8743, 8745, 8747]:
        row = slots[number]
        if row["producer_executed"]:
            value = io.load(row["primary_reference"])
            diagnostics.append(
                dict(
                    experiment_id=number,
                    primary_reference=row["primary_reference"],
                    honest_verdict=value["honest_verdict"],
                    admitted_science=False,
                    gate_check_summary=value.get("gate_check_summary", []),
                    failed_validation_receipts=[
                        r for r in value["validation_receipts"] if not r.get("passed")
                    ],
                )
            )
    blocked["learning"].update(future_brier=None, joint_retention=None)
    return dict(
        **blocked,
        qwen=dict(
            measured=False, support_scores=None, current_cuda_repair=False, language_model_calls=0
        ),
        batch=history["scientific_reconstruction"]["batch"],
        historical_verifier=dict(
            arm_rows=summary["arm_rows"],
            paired_intervals=summary["paired_intervals"],
            gain=summary["matched_call_gain"],
            verdict_class="circular_positive",
            verifier_is_oracle=True,
            exposure_scope="exposed constructed worlds",
            independent_generalization_score=0,
            evaluation_model_calls=0,
            physical_execution=False,
        ),
        arc=arc,
        hardware=dict(
            physical_execution=False,
            tracks=[
                dict(
                    board=board,
                    verdict_class="blocked",
                    physical_execution=False,
                    complete_cost=None,
                )
                for board in ["KV260", "PolarFire", "GateMate"]
            ],
        ),
        blocked_comparison_gates=gates,
        disqualified_diagnostics=diagnostics,
        numerical_training_this_invocation=False,
        current_comparison_count=0,
        continuous_learning_measured=False,
        calibrated_decisions_measured=False,
        arc_generalization_measured=False,
    )
