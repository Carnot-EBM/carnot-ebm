"""REQ-REPORT-8775: reconstruct rows while preserving each producer's eligibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import v746_capstone_inputs as io
from carnot.reporting import v750_capstone_reduce as historical
from carnot.reporting import evidence_set_analysis_8768 as evidence
from carnot.reporting import optimized_cost_attribution_8770 as residual
from carnot.reporting import ordinary_durable_cost_8772 as durable
from carnot.reporting.v747_capstone_reduce import same

Json = dict[str, Any]
HISTORY = dict(
    path=str(io.ROOT / "results/experiment_8762_v750_capstone.json"),
    sha256="sha256:0a2bb03cff8369dfd1a4f8a1d80c1632b1a091e81a20a58e96bfb70d3a8237c4",
)
COMPARISONS = {
    8766: ("learning", "learning_rows_ready_score"),
    8767: ("retention", "learning_benefit_ready_score"),
    8769: ("evidence_set_transfer", "evidence_set_rows_ready_score"),
}


def reduce(work: Json) -> Json:
    """Rejected rows may explain a failure, but they cannot release scientific readiness."""
    history = io.load(HISTORY)
    preserved = historical.reduce(io.load(history["primitive_reference"]))
    same(preserved, history["scientific_reconstruction"], "v750_primitive_reconstruction")
    slots = {r["experiment_id"]: r for r in work["slots"]}
    value: Json = dict(historical_v750=preserved, batch=preserved["batch"])
    gates = []
    for number, (name, field) in COMPARISONS.items():
        row = slots[number]
        gate = io.observe(row["declared_reference"], row["task_id"], field, 1)
        gate.update(producer_qualified=row["producer_qualified"], passed=False)
        gates.append(gate)
        value[name] = dict(
            measured=False,
            verdict_class="blocked",
            rows=[],
            independent_count=0,
            primary_reference=row["primary_reference"],
            admission_gate=gate,
        )
    value["learning"].update(future_brier=None, joint_retention=None)
    for number, name in [
        (8768, "evidence_set_methods"),
        (8770, "optimized_residual_cost"),
        (8772, "bounded_durable_cost"),
    ]:
        row = slots[number]
        primary = io.load(row["primary_reference"])
        primitive = io.load(primary["primitive_reference"])
        if number == 8768:
            inputs, labels, decisions = [
                io.load(primitive[k])["rows"]
                for k in ["corpus_reference", "oracle_reference", "decisions_reference"]
            ]
            rows = evidence.score(inputs, decisions, labels)
            same(rows, primary["rows"], "v751_evidence_decisions")
            summary = evidence.summarize(rows)
            for key in ["arm_summary_rows", "paired_world_rows", "surface_pair_rows"]:
                same(summary[key], primary[key], "v751_evidence_" + key)
            summary.update(independent_count=120, verifier_is_oracle=True)
        elif number == 8770:
            rows = primitive["measurement"]["process_rows"]
            summary = residual.aggregate(rows)
            for key in [
                "component_cost_rows",
                "observer_overhead_rows",
                "reconciliation_rows",
                "ideal_speed_bounds",
            ]:
                same(summary[key], primary[key], "v751_residual_" + key)
            summary.update(independent_count=0)
        else:
            rows = primitive["window_rows"]
            draws = io.load(io.reference(Path(primitive["raw"]) / "bootstrap.json"))["draws"]
            paired = durable.intervals(rows, draws)
            same(paired, primary["paired_repeat_intervals"], "v751_durable_intervals")
            updates = [u for r in rows for u in r["update_rows"]]
            same(rows, primary["per_window_rows"], "v751_durable_windows")
            same(updates, primary["per_update_rows"], "v751_durable_updates")
            summary = dict(
                paired_repeat_intervals=paired,
                completed_window_count=sum(r["status"] == "completed" for r in rows),
                completed_update_count=sum(u["status"] == "completed" for u in updates),
                natural_dense_event_count=sum(
                    u.get("renormalization_writes", 0) > 0 for u in updates
                ),
                durable_commit_count=sum(s["durable_commits"] for r in rows for s in r["segments"]),
                independent_count=1,
            )
            same(
                summary["natural_dense_event_count"],
                primary["natural_dense_event_count"],
                "v751_natural_events",
            )
        value[name] = dict(
            summary,
            reconstructed=True,
            measured=row["producer_qualified"],
            producer_eligible=row["producer_qualified"],
            producer_verdict_class=primary["verdict_class"],
            verdict_class=row["verdict_class"],
            rows=rows,
            primary_reference=row["primary_reference"],
            primitive_reference=primary["primitive_reference"],
            field_provenance={
                k: dict(row["primary_reference"], producer_eligible=row["producer_qualified"])
                for k in summary
            },
            independent_generalization_score=0,
            physical_execution=False,
        )
    value.update(
        blocked_comparison_gates=gates,
        current_comparison_count=0,
        numerical_training_this_invocation=False,
        physical_operations_this_invocation=0,
        continuous_learning_measured=False,
        arc_generalization_measured=False,
    )
    return value
