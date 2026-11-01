"""REQ-REPORT-8695: derive administrative readiness without changing nested operands."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from carnot.reporting import v746_contract_authority as a
from carnot.reporting import v746_retained_evidence as e
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash

Json = dict[str, Any]


def build(work: Json, validation: Json, raw: Path, output: Path) -> Json:
    """Build only from sealed primitives; scientific benefit never follows from registration."""
    w, v = deepcopy(work), e.load(validation)
    receipts = v["receipts"]
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    exact = v["producer_qualification"]["passed"]
    covered = v["owned_coverage"]["complete"]
    mutations = all(r["passed"] for r in w["authority_mutation_rows"])
    qualified = bool(owned and exact and covered and mutations and not w["private_control"])
    contract = bool(qualified and w["contract"]["ready"] and not w["precondition_failures"])
    learning = bool(contract and w["methods_manifest"]["learning_authority_inputs_ready_score"])
    verdict = "disqualified" if not owned else "circular_positive" if learning else "blocked"
    failures = deepcopy(w["precondition_failures"] + w["contract"]["gate_check_summary"])
    failures += w["methods_manifest"]["gate_check_summary"]
    if not learning:
        failures.append(
            a.gate(raw / "validation.json", "exact_owned_learning_authority", True, learning)
        )
    for r in receipts:
        if not r["passed"]:
            failures.append(
                a.gate(Path(r["stdout_path"]), r["name"], r["expected_exit"], r["actual_exit"])
            )
    rows = w["contract"]["rows"]
    value: Json = dict(
        experiment_id=8695,
        task_id=a.TASK,
        milestone=a.MILESTONE,
        run_date=w["run_date"],
        invocation_id=w["invocation_id"],
        honest_verdict=f"complete_{verdict}_v746_retained_contract_authority",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        target_model=dict(
            name="authenticated original nine-input109-coefficient head",
            checkpoint=w["methods_manifest"]["head_checkpoint_reference"],
            evaluated_this_invocation=False,
        ),
        historical_model_provenance=dict(
            scope="Historical provenance only; original numerical validation is separate from current LLM activity.",
            source=w["methods_manifest"]["head_checkpoint_reference"],
        ),
        substrate_by_phase=dict(
            readers="aggregation_from_upstream_artifacts",
            constructed_checks="deterministic_verifier",
            original_numerical_tests="verifier_ensemble_against_cached_candidates",
        ),
        rows=rows,
        intended_count=14,
        completed_count=sum(r["matched"] for r in rows),
        failed_count=0,
        censored_count=sum(not r["matched"] for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            task_slots=14, independent_questions=0, repeats_multiply_evidence=False
        ),
        verifier_is_oracle=True,
        exposure_scope="Constructed administrative controls and original exposed-data methods; no independent natural questions.",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        acceptance_gates=dict(
            owned_validation=owned,
            original_producer_authority=exact,
            newly_owned_statement_coverage=covered,
            authority_mutations=mutations,
            complete_fourteen_objects=w["contract"]["ready"],
            original_learning_inputs=bool(
                w["methods_manifest"]["learning_authority_inputs_ready_score"]
            ),
        ),
        validation_receipts=receipts,
        validation_reference=validation,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=[],
        preconditions_checked=w["preconditions"],
        duration_s=(v["ended_monotonic_ns"] - w["started_monotonic_ns"]) / 1e9,
        started_at_utc=w["started_at_utc"],
        ended_at_utc=datetime.fromtimestamp(v["ended_wall_ns"] / 1e9, UTC).isoformat(),
        phase_spans=w["phase_spans"] + v["phase_spans"],
        random_seed=7468695,
        source_artifact_hashes=w["methods_manifest"]["references"]
        + w["historical_failure"]["references"],
        code_config_hashes=w["source_hashes"],
        raw_shard_hashes=v["raw_references"],
        cited_upstream_artifacts=w["methods_manifest"]["references"],
        branch_gate_values=dict(
            contract_ready_score=int(contract), learning_authority_ready_score=int(learning)
        ),
        contract_ready_score=int(contract),
        learning_authority_ready_score=int(learning),
        retained_child_rows=v["retained_child_rows"],
        producer_coverage_union=v["producer_qualification"],
        original_node_rows=[
            dict(node_id=n, status="completed" if exact else "unqualified", producer_only=True)
            for n in w["original_node_ids"]
        ],
        authority_mutation_rows=w["authority_mutation_rows"],
        methods_manifest=w["methods_manifest"],
        original_rejection_line_executed=v["producer_qualification"][
            "original_rejection_line_executed"
        ],
        original_rejection_source_reference=dict(
            path=str(e.REJECTION_SOURCE), sha256=e.REJECTION_PIN
        ),
        historical_failure_reproduction=w["historical_failure"],
        owned_statement_coverage=v["owned_coverage"],
        historical_quarantines=w["methods_manifest"]["historical_quarantines"],
        preserved_historical_limits=[
            "V733 source-information null",
            "V735 cold-cost null",
            "V737 supervisor null",
            "V743 batch-one null",
            "Exp8681 disqualified",
            "Exp8691 disqualified after later source repair",
            "Exp8692 constructed CPU32/24 only",
            "Exp8693/8694 scientific/device blocks",
        ],
        future_dependencies=w["methods_manifest"]["future_dependencies"],
        physical_fpga_execution=False,
        primitive_reference=e.reference(raw / "measurement.json"),
        publication_output=str(output),
        methodology_note="Collect unchanged original77 nodes once. Retain completed child SQLite, JUnit, exact nodes, source bytes, commands and exits before combining copies. Execute unchanged historical rejection line113 in a distinct producer control. Bind fourteen full V746 task objects to the visible table and independent digest. Qualify owned statement coverage separately from original producer execution and keep every independent branch's original input authority. Zero model loads; no scientific benefit or hardware acceleration claim.",
    )
    value["field_principles"] = {
        k: "Bind this invocation to exact retained evidence; administrative readiness cannot establish independent learning or device execution."
        for k in value
    }
    value["field_principles"]["reproducibility_checksum"] = (
        "Hash the complete report, excluding this checksum."
    )
    value["field_principles"]["field_principles"] = (
        "Explain the evidence boundary of every top-level field."
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value
