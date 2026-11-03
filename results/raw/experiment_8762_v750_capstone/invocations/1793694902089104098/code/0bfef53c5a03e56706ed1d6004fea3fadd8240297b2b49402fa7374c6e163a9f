"""REQ-REPORT-8762: reuse retained reductions with independently bound new operands."""

from __future__ import annotations

from typing import Any

from carnot.reporting import v749_capstone_reduce as historical
from carnot.reporting import v746_capstone_inputs as io
from carnot.reporting import verifier_family_analysis_8755 as verifier
from carnot.reporting import feature_reuse_cost_8757 as cost
from carnot.reporting import feature_reuse_cost_measure_8757 as cost_measure
from carnot.reporting.v747_capstone_reduce import same

Json = dict[str, Any]
HISTORY = dict(
    path=str(io.ROOT / "results/experiment_8748_v749_capstone.json"),
    sha256="sha256:c7892fc565c25d0cef9cbbc242cc2cfbced979a95cbf9c9d51116846ae5c249d",
)
COMPARISONS = {
    8752: ("learning", "learning_rows_ready_score"),
    8753: ("retention", "learning_benefit_ready_score"),
    8755: ("verifier_transfer", "verifier_transfer_rows_ready_score"),
    8757: ("feature_reuse_cost", "installed_cost_ready_score"),
    8759: ("bounded_durable_cost", "ordinary_update_cost_ready_score"),
}


def reduce(work: Json) -> Json:
    """Reconstruct primitive observations; unexecuted branches retain no numerical values."""
    history = io.load(HISTORY)
    value = historical.reduce(io.load(history["primitive_reference"]))
    slots = {r["experiment_id"]: r for r in work["slots"]}
    gates = []
    for number, (name, field) in COMPARISONS.items():
        row = slots[number]
        gate = io.observe(row["declared_reference"], row["task_id"], field, 1)
        gate.update(producer_qualified=row["producer_qualified"])
        gate["passed"] = bool(gate["passed"] and row["producer_qualified"])
        gates.append(gate)
        value[name] = dict(
            measured=False,
            verdict_class="blocked",
            rows=[],
            independent_count=0,
            model_calls=0,
            physical_execution=False,
            primary_reference=row["primary_reference"],
            admission_gate=gate,
        )
    value["learning"].update(future_brier=None, joint_retention=None)
    value["surface_sensitivity"] = dict(measured=False, decision_disagreements=None)
    if slots[8755]["producer_qualified"]:
        primary = io.load(slots[8755]["primary_reference"])
        primitive = io.load(primary["primitive_reference"])
        measured = io.load(primitive["measurement_reference"])
        summary = verifier.summarize(measured["rows"])
        same(
            summary,
            io.load(primitive["analysis_reference"]),
            "v750_verifier_primitive_reconstruction",
        )
        for key in summary:
            if key != "transfer_gain":
                same(summary[key], primary[key], "v750_verifier_" + key)
        value["verifier_transfer"].update(
            summary,
            measured=True,
            verdict_class=primary["verdict_class"],
            rows=measured["rows"],
            independent_count=30,
            primary_probe_calls=sum(len(r["probes"]) for r in measured["rows"]),
            timing_probe_calls=sum(r["calls"] for r in measured["call_cost_rows"]),
            verifier_is_oracle=True,
            independent_generalization_score=0,
            exposure_scope=primary["exposure_scope"],
        )
        value["surface_sensitivity"] = dict(
            measured=True,
            surface_family_rows=summary["surface_family_rows"],
            decision_disagreements=sum(
                r["decision_disagreements"] for r in summary["surface_pair_rows"]
            ),
            truth_discrepancies=sum(r["truth_discrepancies"] for r in summary["surface_pair_rows"]),
            independent_generalization_score=0,
        )
    if slots[8757]["producer_qualified"]:
        primary = io.load(slots[8757]["primary_reference"])
        primitive = io.load(primary["primitive_reference"])
        rows = []
        for row in primitive["measurement"]["process_rows"]:
            session = io.load(row["session_reference"])
            rebuilt = cost_measure.original.reduce(session, primitive["inputs"])["process_row"]
            same({k: row[k] for k in rebuilt}, rebuilt, "v750_complete_process_clock")
            rows.append(row)
        summary = cost.aggregate(rows)
        for key in [
            "paired_process_intervals",
            "native_steady_p95_ms",
            "feature_reuse_benefit_score",
            "same_batch_10x_score",
        ]:
            same(summary[key], primary[key], "v750_cost_" + key)
        value["feature_reuse_cost"].update(
            summary,
            measured=True,
            verdict_class=primary["verdict_class"],
            rows=rows,
            independent_process_repeats=10,
            independent_count=0,
            boundary=primary["inference_boundary"],
            natural_renormalization_frequency=None,
        )
    value.update(
        blocked_comparison_gates=[g for g in gates if not g["passed"]],
        current_comparison_count=sum(value[n]["measured"] for n, _ in COMPARISONS.values()),
        numerical_training_this_invocation=False,
        physical_operations_this_invocation=0,
    )
    return dict(value)
