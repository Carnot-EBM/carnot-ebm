"""REQ-REPORT-8754: readiness requires complete custody and owned validation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from carnot.reporting import verifier_surface_methods_8754 as m
from carnot.reporting.current_work_receipt import canonical_hash, ZERO_INVOCATION_COUNTS

Json = dict[str, Any]


def build(work: Json, validation: Json, output: Path) -> Json:
    """Methods readiness grants no scientific benefit or independent generalization."""
    checked = (
        bool(validation["receipts"])
        and all(r["passed"] for r in validation["receipts"])
        and validation["union"]["passed"]
        and validation["coverage"]["complete"]
        and not work["private_control"]
    )
    gates = dict(
        input_authentication=work["input_ready"],
        truth_equivalence=len(work["truth_equivalence_rows"]) == 480
        and all(r["passed"] for r in work["truth_equivalence_rows"]),
        method_mutations=len(work["method_mutation_rows"]) == 7
        and all(r["passed"] for r in work["method_mutation_rows"]),
        preconditions=work["preconditions"]["private_scratch_writable"]
        and all(work["preconditions"]["tools"].values()),
        owned_validation=checked,
    )
    ready = all(gates.values())
    verdict = "blocked" if not work["input_ready"] else "positive" if ready else "disqualified"
    raw = Path(work["raw"])
    end = validation["ended_at_utc"]
    duration = (
        datetime.fromisoformat(end) - datetime.fromisoformat(work["started_at_utc"])
    ).total_seconds()
    failures = work["gate_check_summary"] + [
        dict(
            upstream=m.TASK,
            **m.reference(raw / "work.json"),
            check=k,
            field=k,
            operator="==",
            expected=True,
            observed=False,
            passed=False,
        )
        for k, v in gates.items()
        if not v
    ]
    rows = work["truth_equivalence_rows"]
    value = dict(
        experiment_id=8754,
        task_id=m.TASK,
        milestone="2026.11.750",
        run_date="20261103",
        invocation_id=work["invocation_id"],
        honest_verdict="complete_" + verdict + "_v750_verifier_surface_methods",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        rows=rows,
        intended_count=480,
        completed_count=len(rows),
        failed_count=sum(not r["passed"] for r in rows),
        censored_count=480 - len(rows),
        excluded_count=0,
        independent_count=30 if work["input_ready"] else 0,
        verifier_is_oracle=False,
        exposure_scope="Constructed V749 inputs and labels already inspected; exposed evaluation partition, never blind held-out data. No verifier executed in this task.",
        independent_generalization_score=0,
        acceptance_gates=gates,
        required_checks_passed=checked,
        flagged_adversarial=False,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["preconditions"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=end,
        duration_s=duration,
        phase_spans=[
            dict(
                phase="authentication_and_freeze",
                start_utc=work["started_at_utc"],
                end_utc=work["frozen_at_utc"],
            ),
            dict(phase="scoped_qualification", start_utc=work["frozen_at_utc"], end_utc=end),
        ],
        random_seed=m.SEED,
        source_artifact_hashes=work["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[
            m.reference(raw / "work.json"),
            m.reference(raw / "validation.json"),
            *work["workload_references"],
        ],
        cited_upstream_artifacts=work["references"],
        repository_health_reference=work["repository_health_reference"],
        scoped_validation_scope=dict(
            required_suites=m.TESTS + m.CONSUMERS,
            exact_node_ids=validation["union"].get("node_ids", []),
            repository_health_is_dependency=False,
            original_assertions_unchanged=True,
            e2e=["E2E-018", "E2E-021"],
        ),
        verifier_transfer_methods_ready_score=int(ready),
        verifier_protocol_reference=work["verifier_protocol_reference"],
        surface_transform_reference=work["surface_transform_reference"],
        truth_equivalence_rows=rows,
        method_mutation_rows=work["method_mutation_rows"],
        canonical_workload=work["canonical_workload"],
        transformed_workload=work["transformed_workload"],
        analysis_protocol=work["analysis_protocol"],
        scientific_promotion_gates=dict(
            measured=False,
            oracle_dependent_benefit_verdict="circular_positive",
            independent_generalization=False,
        ),
        primitive_reference=m.reference(raw / "work.json"),
        validation_reference=m.reference(raw / "validation.json"),
        publication_output=str(output.absolute()),
        methodology_note="Authenticate retained V749 protocol as input only. Independent signed-fact enumeration checks480 paired probes. Seed7508754 freezes one entity bijection and premise-order permutation per world. Preserve60 constructed worlds,30 exposed evaluation clusters, six variants and two fixed five-call schedules. Freeze720 rows/3600 calls per representation; no verifier is executed. Solver-Hard motivates representation controls, not a solver-effort quality proxy. Correct alternatives, falsehood and unknown evidence remain distinct. Complete original archived tests and private readers qualify methods independently of unrelated repository health. No LLM, training, learned verifier or scientific gain is measured.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Bind exact producer bytes and actual clocks; paired surface forms preserve truth and sample identity; execution qualification is distinct from scientific promotion.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value
