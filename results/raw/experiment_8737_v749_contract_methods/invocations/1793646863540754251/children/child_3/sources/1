"""REQ-REPORT-8737: administrative readiness never becomes scientific evidence."""

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from carnot.reporting import v749_contract_methods as m
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash

Json = dict[str, Any]


def build(work: Json, validation_ref: Json, output: Path) -> Json:
    """Reduce sealed method operands and actual checks into independent branch gates."""
    validation = m.load(validation_ref)
    raw = Path(work["raw"])
    checked = validation["passed"] and validation["coverage"]["complete"]
    gates = dict(
        contract=work["contract"]["ready"],
        learning_inputs=work["learning"]["ready"],
        upstream_custody=work["upstream"]["ready"],
        verifier_schema=len(work["verifier"]["rows"]) == 720
        and len(work["verifier"]["worlds"]) == 60,
        mutations=all(r["passed"] for r in work["mutations"]),
        exclusion=work["exclusion"]["passed"],
        owned_validation=checked,
    )
    external = (
        gates["contract"]
        and gates["learning_inputs"]
        and gates["upstream_custody"]
        and gates["exclusion"]
    )
    verdict = "blocked" if not external else "positive" if all(gates.values()) else "disqualified"
    failures = [
        dict(
            upstream=m.TASK,
            path=str(raw / "work.json"),
            sha256=m.reference(raw / "work.json")["sha256"],
            check=name,
            field=name,
            operator="==",
            expected=True,
            observed=passed,
            passed=False,
        )
        for name, passed in gates.items()
        if not passed
    ]
    failures += [
        dict(
            upstream=m.TASK,
            path=r["stdout_path"],
            sha256=r["stdout_sha256"],
            check=r["name"],
            field="exit_code",
            operator="==",
            expected=r["expected_exit"],
            observed=r["exit_code"],
            timed_out=r["timed_out"],
            passed=False,
        )
        for r in validation["receipts"]
        if not r["passed"] and "stdout_sha256" in r
    ]
    failures += work["learning"].get("gate_check_summary", []) + work["upstream"].get(
        "gate_check_summary", []
    )
    end = validation.get("ended_at_utc", work["ended_at_utc"])
    duration = (
        datetime.fromisoformat(end) - datetime.fromisoformat(work["started_at_utc"])
    ).total_seconds()
    value = dict(
        experiment_id=8737,
        task_id=m.TASK,
        milestone=m.MILESTONE,
        run_date="20261102",
        invocation_id=work["invocation_id"],
        honest_verdict="complete_" + verdict + "_v749_contract_methods",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="upstream_artifact_aggregation",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        rows=work["contract"]["rows"],
        intended_count=12,
        completed_count=len(work["contract"]["rows"]),
        failed_count=0,
        censored_count=12 - len(work["contract"]["rows"]),
        excluded_count=0,
        independent_count=0,
        verifier_is_oracle=False,
        exposure_scope="administrative registration of constructed oracle-dependent methods; no new science",
        independent_generalization_score=0,
        acceptance_gates=gates,
        required_checks_passed=checked,
        flagged_adversarial=False,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["exclusion"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=end,
        duration_s=duration,
        phase_spans=[
            dict(
                phase="methods_and_validation",
                start_utc=work["started_at_utc"],
                end_utc=end,
                duration_s=duration,
            )
        ],
        random_seed=7498737,
        source_artifact_hashes=work["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[
            m.reference(raw / "work.json"),
            validation_ref,
            work["learning_protocol_reference"],
            work["verifier_transfer_protocol_reference"],
            work["cost_protocol_reference"],
        ],
        cited_upstream_artifacts=work["upstream"]["references"],
        contract_ready_score=int(gates["contract"] and gates["mutations"] and checked),
        learning_protocol_ready_score=int(
            gates["learning_inputs"] and gates["upstream_custody"] and checked
        ),
        verifier_transfer_methods_ready_score=int(gates["verifier_schema"] and checked),
        learning_execution_ready_score=0,
        full_task_objects=work["contract"]["tasks"],
        visible_task_table=work["contract"]["visible_table"],
        canonical_tasks_sha256=m.PIN,
        preserved_v748_canonical_tasks_sha256=m.historical.m.PIN,
        contract_mutation_rows=work["mutations"],
        learning_protocol_reference=work["learning_protocol_reference"],
        verifier_transfer_protocol_reference=work["verifier_transfer_protocol_reference"],
        cost_protocol_reference=work["cost_protocol_reference"],
        literature_method_rows=work["literature"],
        upstream_authentication=work["upstream"],
        learning_execution_authorized=False,
        new_scientific_measurements=0,
        intended_scientific_units=dict(
            worlds=60,
            world_variant_arm_rows=720,
            unique_probe_invocations=3600,
            timed_repetitions=10,
            evaluation_defective_units_per_arm=150,
            evaluation_new_family_units_per_arm=60,
        ),
        publication_output=str(output.absolute()),
        primitive_reference=m.reference(raw / "work.json"),
        validation_reference=validation_ref,
        methodology_note="Bind twelve complete V749 tasks and separately freeze original V744 numerical methods, independent constructed truth and unchanged five-call schedules. Authenticate raw passed core children without promoting failed smoke or old primary. No model execution, new outcomes, learning benefit, independent generalization, CUDA repair, ARC policy change or physical acceleration. Method readiness does not authorize learning execution.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Bind the actual producer and sealed evidence; keep method eligibility separate from execution and scientific benefit.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value
