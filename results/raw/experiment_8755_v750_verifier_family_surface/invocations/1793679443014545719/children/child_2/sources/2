"""REQ-REPORT-8755: execution acceptance and scientific promotion have distinct gates."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from carnot.reporting import verifier_family_surface_8755 as m
from carnot.reporting import verifier_family_analysis_8755 as analysis
from carnot.reporting import verifier_surface_report_8754 as original
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]


def build(work: Json, validation: Json, output: Path) -> Json:
    """A null is a completed measurement; an owned execution failure cannot qualify."""
    value = original.build(work, validation, output)
    value.pop("verifier_transfer_methods_ready_score")
    measured = m.load(work["measurement_reference"])
    rows = measured["rows"]
    summary = analysis.summarize(rows)
    if summary != m.load(work["analysis_reference"]):
        raise ValueError("sealed_analysis_drift")
    gain = summary.pop("transfer_gain")
    gates = dict(
        value["acceptance_gates"],
        complete_measurement=len(rows) == 1440 and len(measured["call_cost_rows"]) == 14400,
        independent_row_audit=work["row_audit_passed"],
        numerical_budget=(
            datetime.fromisoformat(work["numerical_ended_at_utc"])
            - datetime.fromisoformat(work["started_at_utc"])
        ).total_seconds()
        <= 2400,
    )
    ready = all(gates.values())
    verdict = (
        "blocked"
        if not work["input_ready"]
        else "disqualified"
        if not ready
        else "circular_positive"
        if gain
        else "null"
    )
    intended = m.roster(
        {s: m.load(work[s + "_workload"]["reference"]) for s in ["canonical", "transformed"]}
    )
    unfinished = (
        []
        if rows
        else [
            dict(
                **row,
                status="blocked_external_operand" if verdict == "blocked" else "censored",
                absolute_metric=0,
                censored=True,
                excluded=False,
            )
            for row in intended
        ]
    )
    value.update(
        summary,
        experiment_id=8755,
        task_id=m.TASK,
        honest_verdict="complete_" + verdict + "_v750_verifier_family_surface",
        verdict_class=verdict,
        rows=rows or unfinished,
        intended_count=1440,
        completed_count=len(rows),
        failed_count=0,
        censored_count=1440 - len(rows),
        independent_count=30 if rows else 0,
        independent_constructed_worlds=60 if rows else 0,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        verifier_is_oracle=True,
        verifier_transfer_rows_ready_score=int(ready),
        acceptance_gates=gates,
        call_cost_rows=measured["call_cost_rows"],
        primary_probe_invocations=sum(len(r["probes"]) for r in rows),
        timing_probe_invocations=sum(r["calls"] for r in measured["call_cost_rows"]),
        equal_wall_time_advantage_claimed=False,
        numerical_training_calls=0,
        random_seed=m.SEED,
        exposure_scope="Sixty constructed and previously exposed worlds; thirty original canonical evaluation clusters. Surface siblings, variants and ten timing repeats add no independent science units.",
        primitive_reference=m.reference(Path(work["raw"]) / "work.json"),
        validation_reference=m.reference(Path(work["raw"]) / "validation.json"),
        raw_shard_hashes=[
            m.reference(Path(work["raw"]) / "work.json"),
            m.reference(Path(work["raw"]) / "validation.json"),
            work["measurement_reference"],
            work["analysis_reference"],
            *work["workload_references"],
        ],
        scoped_validation_scope=dict(
            required_suites=m.TESTS + m.CONSUMERS,
            exact_node_ids=validation["union"].get("node_ids", []),
            frozen_node_reference=work["frozen_node_reference"],
            repository_health_is_dependency=False,
            original_assertions_unchanged=True,
            e2e=["E2E-018", "E2E-021"],
        ),
        methodology_note="Execute both sealed Exp8754 workloads with six fixed variants and two unchanged five-call schedules. Reuse the original exhaustive signed-fact evaluator; independent enumeration checks labels. Primary transfer uses30 original canonical evaluation worlds,150 defective units per arm including60 new-family units, and10000 paired world-cluster draws. Paired surfaces expose representation shortcuts separately. Ten complete five-call timing repeats measure cost only. Predicate work counts supplied query predicates, not machine instructions. Correct alternatives and missing-premise/falsehood controls stay distinct. A fixed-study null closes the study; oracle benefit remains circular_positive. No LLM, training, natural extraction or independent generalization is established.",
    )
    value["gate_check_summary"] += [
        dict(
            upstream=m.TASK,
            **work["measurement_reference"],
            check=k,
            field=k,
            operator="==",
            expected=True,
            observed=False,
            passed=False,
        )
        for k in ["complete_measurement", "independent_row_audit"]
        if not gates[k]
    ]
    value["phase_spans"] = [
        dict(phase=name, start_utc=work[start], end_utc=end)
        for name, start, end in [
            ("freeze", "started_at_utc", work["frozen_at_utc"]),
            ("numerical", "numerical_started_at_utc", work["numerical_ended_at_utc"]),
            ("validation", "numerical_ended_at_utc", validation["ended_at_utc"]),
        ]
    ]
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles"],
        "Bind this invocation's sealed inputs, exact calls and clocks; thirty exposed canonical clusters define primary science; cost repeats and surfaces never enlarge sample size; execution acceptance is separate from transfer and invariance.",
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value
